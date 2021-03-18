# Written by Yixiao Ge

import collections
import os
import os.path as osp
import time
import warnings
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
try:
    # PyTorch >= 1.6 supports mixed precision training
    from torch.cuda.amp import GradScaler, autocast
    amp_support = True
except:
    amp_support = False

from ..core.label_generators import LabelGenerator
from ..core.metrics.accuracy import accuracy
from ..data import build_train_dataloader, build_val_dataloader, build_test_dataloader
from ..data.utils.data_utils import save_image
from ..utils import bcolors
from ..utils.dist_utils import get_dist_info, synchronize
from ..utils.meters import Meters
from ..utils.image_pool import ImagePool
from ..utils.torch_utils import copy_state_dict, load_checkpoint, save_checkpoint
from ..utils.file_utils import mkdir_if_missing
from ..utils.torch_utils import tensor2im
from .test import val_reid, test_reid
from .train import batch_processor, set_random_seed
from openunreid.data.utils.data_utils import save_image

class BaseRunner(object):
    """
    Re-ID Base Runner
    """

    def __init__(
        self,
        cfg,
        model,
        optimizer,
        criterions,
        train_loader,
        train_sets=None,
        lr_scheduler=None,
        meter_formats=None,
        print_freq=10,
        reset_optim=True,
        label_generator=None,
    ):
        super(BaseRunner, self).__init__()
        # set_random_seed(cfg.TRAIN.seed, cfg.TRAIN.deterministic)

        if meter_formats is None:
            meter_formats = {"Time": ":.3f", "Acc@1": ":.2%"}

        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.criterions = criterions
        self.lr_scheduler = lr_scheduler
        self.print_freq = print_freq
        self.reset_optim = reset_optim
        self.label_generator = label_generator

        self.is_pseudo = (
            "PSEUDO_LABELS" in self.cfg.TRAIN
            and self.cfg.TRAIN.unsup_dataset_indexes is not None
        )
        if self.is_pseudo:
            if self.label_generator is None:
                self.label_generator = LabelGenerator(self.cfg, self.model["id"])

        self._rank, self._world_size, self._is_dist = get_dist_info()
        self._epoch, self._start_epoch = 0, 0
        self._best_mAP = 0
        self.alpha = 10
        self.mix_weight = 0.5

        # build data loaders
        self.train_loader, self.train_sets = train_loader, train_sets
        if "val_dataset" in self.cfg.TRAIN:
            self.val_loader, self.val_set = build_val_dataloader(cfg)
        self.test_loaders, self.queries, self.galleries = build_test_dataloader(cfg)

        # save training variables
        for key in criterions.keys():
            meter_formats[key] = ":.3f"
        self.train_progress = Meters(
            meter_formats, self.cfg.TRAIN.iters, prefix="Train: "
        )

        # build mixed precision scaler
        if "amp" in cfg.TRAIN:
            global amp_support
            if cfg.TRAIN.amp and amp_support:
                assert not isinstance(model, DataParallel), \
                    "We do not support mixed precision training with DataParallel currently"
                self.scaler = GradScaler()
            else:
                if cfg.TRAIN.amp:
                    warnings.warn(
                        "Please update the PyTorch version (>=1.6) to support mixed precision training"
                    )
                self.scaler = None
        else:
            self.scaler = None

        self.save_dir = osp.join(self.cfg.work_dir, 'images')
        if self._rank == 0:
            mkdir_if_missing(self.save_dir)

    def run(self):
        for i, (loader, query, gallery) in enumerate(zip(self.test_loaders, self.queries, self.galleries)):
            cmc, mAP = test_reid(
                self.cfg, self.model['id'], loader, query, gallery, dataset_name=self.cfg.TEST.datasets[i]
            )
        self.save(mAP)
        # the whole process for training
        for ep in range(self._start_epoch, self.cfg.TRAIN.epochs):
            self._epoch = ep

            # loss_id = self.train_progress.meters['spcl'].avg+self.train_progress.meters['co'].avg
            # if loss_id != 0:
            #     # self.mix_weight = torch.exp(-torch.tensor([loss_id])).cuda()
            #     self.mix_weight = min(np.exp(-loss_id), 0.5)
            #     print(self.mix_weight)

            # generate pseudo labels
            if self.is_pseudo:
                if (
                    ep % self.cfg.TRAIN.PSEUDO_LABELS.freq == 0
                    or ep == self._start_epoch
                ):
                    self.update_labels()
                    synchronize()

            # train
            self.train()
            synchronize()

            # validate
            # if (ep + 1) % self.cfg.TRAIN.val_freq == 0 or (
            #     ep + 1
            # ) == self.cfg.TRAIN.epochs:
            #     if "val_dataset" in self.cfg.TRAIN:
            #         mAP = self.val()
            #         # self.save(mAP)
            #     else:
            #         self.save()

            for i, (loader, query, gallery) in enumerate(zip(self.test_loaders, self.queries, self.galleries)):
                cmc, mAP = test_reid(
                    self.cfg, self.model['id'], loader, query, gallery, dataset_name=self.cfg.TEST.datasets[i]
                )
            self.save(mAP)

            # update learning rate
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, list):
                    for scheduler in self.lr_scheduler:
                        scheduler.step()
                elif isinstance(self.lr_scheduler, dict):
                    for key in self.lr_scheduler.keys():
                        self.lr_scheduler[key].step()
                else:
                    self.lr_scheduler.step()

            # synchronize distributed processes
            synchronize()

    def update_labels(self):
        sep = "*************************"
        print(f"\n{sep} Start updating pseudo labels on epoch {self._epoch} {sep}\n")

        # generate pseudo labels
        pseudo_labels, label_centers = self.label_generator(
            self._epoch, print_freq=self.print_freq
        )

        self.model['id'].module.num_classes = label_centers[0].size(0)
        self.model['id'].module.add_module('classifier', nn.Linear(self.model['id'].module.num_features,
                                                                   self.model['id'].module.num_classes, bias=False))
        self.model['id'].module.classifier.cuda()

        # update train loader
        self.train_loader, self.train_sets = build_train_dataloader(
            self.cfg, pseudo_labels, self.train_sets, self._epoch,
        )
        # os.mkdir('/home/dongdong_li/OpenUnReID/logs/strong_baseline/market/cluster')
        # for i in list(range(self.model['id'].module.num_classes)):
        #     path = '/home/dongdong_li/OpenUnReID/logs/strong_baseline/market/cluster/' + str(i)
        #     os.mkdir(path)
        #     for t in self.train_sets[0].data:
        #         if t[1] == i:
        #             image = Image.open(t[0])
        #             img = np.array(image)
        #             save_image(img, osp.join(path, t[0].split('/')[-1]))


        # update criterions
        if "cross_entropy" in self.criterions.keys():
            self.criterions[
                "cross_entropy"
            ].num_classes = self.train_loader.loader.dataset.num_pids

        # reset optim (optional)
        if self.reset_optim:
            self.optimizer["id"].state = collections.defaultdict(dict)

        # # update classifier centers
        # start_cls_id = 0
        # for idx in range(len(self.cfg.TRAIN.datasets)):
        #     if idx in self.cfg.TRAIN.unsup_dataset_indexes:
        #         labels = torch.arange(
        #             start_cls_id, start_cls_id + self.train_sets[idx].num_pids
        #         )
        #         centers = label_centers[self.cfg.TRAIN.unsup_dataset_indexes.index(idx)]
        #         if isinstance(self.model, list):
        #             for model in self.model:
        #                 if isinstance(model, (DataParallel, DistributedDataParallel)):
        #                     model = model.module
        #                 model.initialize_centers(centers, labels)
        #         else:
        #             model = self.model
        #             if isinstance(model, (DataParallel, DistributedDataParallel)):
        #                 model = model.module
        #             model.initialize_centers(centers, labels)
        #     start_cls_id += self.train_sets[idx].num_pids

        # update classifier centers
        start_cls_id = 0
        for idx in range(len(self.cfg.TRAIN.datasets)):
            if idx in self.cfg.TRAIN.unsup_dataset_indexes:
                labels = torch.arange(
                    start_cls_id, start_cls_id + self.train_sets[idx].num_pids
                )
                centers = label_centers[self.cfg.TRAIN.unsup_dataset_indexes.index(idx)]
                self.model['id'].module.initialize_centers(centers, labels)
            start_cls_id += self.train_sets[idx].num_pids

        print(f"\n{sep} Finished updating pseudo label {sep}n")

    def train(self):
        # one loop for training
        if isinstance(self.model, list):
            for model in self.model:
                model.train()
        elif isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].train()
        else:
            self.model.train()

        if self.cfg.TRAIN.stage == 2 or (self.cfg.TRAIN.stage==3 and not self.cfg.TRAIN.stage3_train_all):
            self.model['id'].eval()

        self.train_progress.reset(prefix="Epoch: [{}]".format(self._epoch))

        if isinstance(self.train_loader, list):
            for loader in self.train_loader:
                loader.new_epoch(self._epoch)
        else:
            self.train_loader.new_epoch(self._epoch)

        end = time.time()
        for iter in range(self.cfg.TRAIN.iters):

            if isinstance(self.train_loader, list):
                batch = [loader.next() for loader in self.train_loader]
            else:
                batch = self.train_loader.next()

            if self.scaler is None:
                if self.cfg.TRAIN.stage == 2:
                    loss_D,loss_recon = self.train_step(iter, batch)
                    meters = {}
                    meters['gan_D'] = loss_D.item()
                    meters['recon'] = loss_recon.item()
                    self.train_progress.update(meters)
                elif self.cfg.TRAIN.stage == 3:
                    spcl_loss,co_loss,ad_loss,loss_D,loss_recon,ad_ce_loss= self.train_step(iter, batch)
                    meters = {}
                    # meters['hybrid_memory'] = loss_all.item()
                    if spcl_loss != 0:
                        meters['spcl'] = spcl_loss.item()
                    meters['co'] = co_loss.item()
                    if ad_loss != 0:
                        meters['ad'] = ad_loss.item()
                    meters['gan_D'] = loss_D.item()
                    meters['recon'] = loss_recon.item()
                    if ad_ce_loss != 0:
                        if 'ad_cross_entropy' in meters.keys():
                            meters['ad_cross_entropy'] = ad_ce_loss.item()
                        else:
                            meters['cross_entropy'] = ad_ce_loss.item()
                    self.train_progress.update(meters)
            else:
                with autocast():
                    loss = self.train_step(iter, batch)
                if (loss > 0):
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)

            if self.scaler is not None:
                self.scaler.update()

            self.train_progress.update({"Time": time.time() - end})
            end = time.time()

            if iter % self.print_freq == 0:
                self.train_progress.display(iter)

    def save_imgs(self, input):
        for name,imgs in input.items():
            for i,img in enumerate(imgs):
                # img = img[0]
                img_np = tensor2im(img, mean=self.cfg.DATA.norm_mean, std=self.cfg.DATA.norm_std)
                save_image(img_np, osp.join(self.save_dir, 'epoch_{}_{}_{}.jpg'.format(self._epoch, name, i)))

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def train_step(self, iter, batch):
        # need to be re-written case by case
        assert not isinstance(
            self.model, list
        ), "please re-write 'train_step()' to support list of models"

        data = batch_processor(batch, self.cfg.MODEL.dsbn)
        if len(data["img"]) > 1:
            warnings.warn(
                "please re-write the 'runner.train_step()' function to make use of "
                "mutual transformer."
            )

        if isinstance(data["img"],list):
            x_ori = data["img"][0].cuda()
            x_aug = data["img"][1].cuda()
        else:
            inputs = data["img"].cuda()
        targets = data["id"].cuda()
        indexes = data["ind"].cuda()
        # inputs = data["img"][0].cuda()
        # targets = data["id"].cuda()

        index = list(range(len(targets)))
        #dic_target = dict(zip(targets.tolist(), index))
        dic_target = {}
        for k, v in zip(targets.tolist(), index):
            if k in dic_target.keys():
                dic_target[k].append(v)
            else:
                dic_target[k] = [v]
        x_neg = []
        targets_neg = []
        for i, target in enumerate(targets):
            neg_targets = set(targets.tolist()) - {target.item()}
            neg_target = np.random.choice(list(neg_targets))
            #neg_index = dic_target[neg_target]
            neg_index = np.random.choice(dic_target[neg_target])
            x_neg.append(x_ori[neg_index])
            targets_neg.append(neg_target)
        x_neg = torch.stack(x_neg)
        targets_neg = torch.tensor(targets_neg)

        # results = self.model(inputs)
        # if "prob" in results.keys():
        #     results["prob"] = results["prob"][
        #         :, : self.train_loader.loader.dataset.num_pids
        #     ]

        x_gen_ori, x_gen_ori_neg = self.model['generator'](x_ori, x_neg, self.mix_weight)

        if self.cfg.TRAIN.stage == 2:
            # -------------------------------------------
            self.set_requires_grad([self.model['discriminator']], False)
            self.optimizer['generator'].zero_grad()
            # d_x_gen_ori_neg = self.model['discriminator'](x_gen_ori_neg)
            # d_x_gen_ori = self.model['discriminator'](x_gen_ori)
            # loss_G_real1 = self.criterions['gan_G'](d_x_gen_ori_neg, True)
            # loss_G_real2 = self.criterions['gan_G'](d_x_gen_ori, True)
            # loss_G_real = (loss_G_real1 + loss_G_real2)
            loss_recon = self.criterions['recon'](x_gen_ori, x_ori) * 10
            loss_G = loss_recon
            # loss_G = loss_recon+ad_loss
            loss_G.backward(retain_graph=False)
            self.optimizer['generator'].step()

            #-------------------------------------------
            self.set_requires_grad([self.model['discriminator']], True)
            self.optimizer['discriminator'].zero_grad()
            d_x_ori = self.model['discriminator'](x_ori)
            d_x_gen_ori_neg = self.model['discriminator'](x_gen_ori_neg.detach())
            d_x_gen_ori = self.model['discriminator'](x_gen_ori.detach())
            loss_D_real = self.criterions['gan_D'](d_x_ori, True)
            loss_D_fake1 = self.criterions['gan_D'](d_x_gen_ori_neg, False)
            loss_D_fake2 = self.criterions['gan_D'](d_x_gen_ori, False)
            loss_D = (loss_D_real + loss_D_fake1 + loss_D_fake2)/3
            loss_D.backward()
            self.optimizer['discriminator'].step()

        if self.cfg.TRAIN.stage == 3:
            results, backbone_ori = self.model['id'](x_ori, val=False)

            results_gen, backbone_gen = self.model['id'](x_gen_ori_neg,val=False)
            results["pooling_gen"] = results_gen['pooling']
            results["feat_gen"] = results_gen['feat']
            results["prob_gen"] = results_gen['prob']

            results_aug, backbone_aug = self.model['id'](x_aug,val=False)
            results["pooling_aug"] = results_aug['pooling']
            results["feat_aug"] = results_aug['feat']
            results["prob_aug"] = results_aug['prob']

            ce_loss = self.criterions["cross_entropy"](results_gen, targets_neg) * 1
            # ce_loss = 0

            loss_id = 0
            meters = {}
            if "prob_aug" in results.keys():
                acc = accuracy(results["prob_aug"].data, targets.data)
                meters["Acc@1"] = acc[0]
            for key in self.criterions.keys():
                if key in ['co_ad',"recon","gan_D", "co" ,"ad","ad_cross_entropy"]:
                    continue
                else:
                    loss = self.criterions[key](results_aug, targets)
                loss_id += loss * float(self.cfg.TRAIN.LOSS.losses[key])
                meters[key] = loss.item()

            self.train_progress.update(meters)

            ad_loss, co_loss = self.criterions["co_ad"](results, targets)
            loss_id += co_loss

            if self.cfg.TRAIN.stage3_train_all:
                self.model['id'].train()
                self.model['generator'].eval()
                self.model['discriminator'].eval()

                self.optimizer['id'].zero_grad()
                loss_id.backward(retain_graph=True)
                self.optimizer['id'].step()

                self.model['generator'].train()
                self.model['discriminator'].train()
                self.model['id'].eval()

            self.set_requires_grad([self.model['discriminator']], False)
            self.optimizer['generator'].zero_grad()
            loss_recon = self.criterions['recon'](x_gen_ori, x_ori) * 1
            loss_G = loss_recon + ad_loss + ce_loss
            loss_G.backward(retain_graph=False)
            self.optimizer['generator'].step()

            self.set_requires_grad([self.model['discriminator']], True)
            self.optimizer['discriminator'].zero_grad()
            d_x_ori = self.model['discriminator'](x_ori)
            d_x_gen_ori_neg = self.model['discriminator'](x_gen_ori_neg.detach())
            d_x_gen_ori = self.model['discriminator'](x_gen_ori.detach())
            loss_D_real = self.criterions['gan_D'](d_x_ori, True)
            loss_D_fake1 = self.criterions['gan_D'](d_x_gen_ori_neg, False)
            loss_D_fake2 = self.criterions['gan_D'](d_x_gen_ori, False)
            loss_D = (loss_D_real + loss_D_fake1 + loss_D_fake2) / 3
            loss_D.backward()
            self.optimizer['discriminator'].step()

            # print('spcl_loss', spcl_loss.item(), 'co_loss:', co_loss.item(), ' ad_loss:', ad_loss.item())

        # save translated images
        if self._rank == 0:
            self.save_imgs({'x_ori': x_ori, 'x_neg': x_neg, 'x_gen_ori_neg': x_gen_ori_neg, 'x_gen_ori': x_gen_ori})

        if self.cfg.TRAIN.stage == 2:
            return loss_D,loss_recon
        elif self.cfg.TRAIN.stage == 3:
            return 0,co_loss,ad_loss,loss_D,loss_recon,ce_loss

        # total_loss = 0
        # meters = {}
        # for key in self.criterions.keys():
        #     loss = self.criterions[key](results, targets)
        #     total_loss += loss * float(self.cfg.TRAIN.LOSS.losses[key])
        #     meters[key] = loss.item()
        #
        # if "prob" in results.keys():
        #     acc = accuracy(results["prob"].data, targets.data)
        #     meters["Acc@1"] = acc[0]
        #
        # self.train_progress.update(meters)
        #
        # return total_loss

    def val(self):
        if not isinstance(self.model, list):
            model_list = [self.model]
        else:
            model_list = self.model

        better_mAP = 0
        for idx in range(len(model_list)):
            if len(model_list) > 1:
                print("==> Val on the no.{} model".format(idx))
            cmc, mAP = val_reid(
                self.cfg,
                model_list[idx]['id'],
                self.val_loader[0],
                self.val_set[0],
                self._epoch,
                self.cfg.TRAIN.val_dataset,
                self._rank,
                print_freq=self.print_freq,
            )
            better_mAP = max(better_mAP, mAP)

        return better_mAP

    def save(self, mAP=None):
        if mAP is not None:
            is_best = mAP > self._best_mAP
            self._best_mAP = max(self._best_mAP, mAP)
            print(
                bcolors.OKGREEN
                + "\n * Finished epoch {:3d}  mAP: {:5.1%}  best: {:5.1%}{}\n".format(
                    self._epoch, mAP, self._best_mAP, " *" if is_best else ""
                )
                + bcolors.ENDC
            )
        else:
            is_best = True
            print(
                bcolors.OKGREEN
                + "\n * Finished epoch {:3d} \n".format(self._epoch)
                + bcolors.ENDC
            )

        if self._rank == 0:
            # only on cuda:0
            self.save_model(is_best, self.cfg.work_dir)

    def save_model(self, is_best, fpath):
        if isinstance(self.model, list):
            state_dict = {}
            state_dict["epoch"] = self._epoch + 1
            state_dict["best_mAP"] = self._best_mAP
            for idx, model in enumerate(self.model):
                state_dict["state_dict_" + str(idx + 1)] = model.state_dict()
            save_checkpoint(state_dict, is_best,
                    fpath=osp.join(fpath, "checkpoint.pth"))

        elif isinstance(self.model, dict):
            state_dict = {}
            state_dict["epoch"] = self._epoch + 1
            state_dict["best_mAP"] = self._best_mAP
            for key in self.model.keys():
                state_dict["state_dict"] = self.model[key].state_dict()
                save_checkpoint(state_dict, is_best, key,
                        fpath=osp.join(fpath, "checkpoint_"+key+".pth"))

        elif isinstance(self.model, nn.Module):
            state_dict = {}
            state_dict["epoch"] = self._epoch + 1
            state_dict["best_mAP"] = self._best_mAP
            state_dict["state_dict"] = self.model.state_dict()
            save_checkpoint(state_dict, is_best,
                        fpath=osp.join(fpath, "checkpoint.pth"))

        else:
            assert "Unknown type of model for save_model()"

    def resume(self, path):
        # resume from a training checkpoint (not source pretrain)
        self.load_model(path)
        synchronize()

    def load_model(self, path):
        if isinstance(self.model, list):
            assert osp.isfile(path)
            state_dict = load_checkpoint(path)
            for idx, model in enumerate(self.model):
                copy_state_dict(state_dict["state_dict_" + str(idx + 1)], model)

        elif isinstance(self.model, dict):
            if osp.isfile(path):
                state_dict = load_checkpoint(path)
                copy_state_dict(state_dict["state_dict"], self.model['id'])
            else:
                assert osp.isdir(path)
                for key in self.model.keys():
                    state_dict = load_checkpoint(osp.join(path, "checkpoint_"+key+".pth"))
                    copy_state_dict(state_dict["state_dict"], self.model[key])

        elif isinstance(self.model, nn.Module):
            assert osp.isfile(path)
            state_dict = load_checkpoint(path)
            copy_state_dict(state_dict["state_dict"], self.model)

        self._start_epoch = state_dict["epoch"]
        self._best_mAP = state_dict["best_mAP"]

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size


class GANBaseRunner(BaseRunner):
    """
    Domain-translation Base Runner
    Re-implementation of CycleGAN
    """

    def __init__(
        self,
        cfg,
        model,
        optimizer,
        criterions,
        train_loader,
        **kwargs
    ):
        super(GANBaseRunner, self).__init__(
            cfg, model, optimizer, criterions, train_loader, **kwargs
        )

        self.save_dir = osp.join(self.cfg.work_dir, 'images')
        if self._rank == 0:
            mkdir_if_missing(self.save_dir)

        self.fake_A_pool = ImagePool()
        self.fake_B_pool = ImagePool()

    def train_step(self, iter, batch):
        data_src, data_tgt = batch[0], batch[1]

        self.real_A = data_src['img'].cuda()
        self.real_B = data_tgt['img'].cuda()

        # Forward
        self.fake_B = self.model['G_A'](self.real_A)     # G_A(A)
        self.fake_A = self.model['G_B'](self.real_B)     # G_B(B)
        self.rec_A = self.model['G_B'](self.fake_B)    # G_B(G_A(A))
        self.rec_B = self.model['G_A'](self.fake_A)    # G_A(G_B(B))

        # G_A and G_B
        self.set_requires_grad([self.model['D_A'], self.model['D_B']], False) # save memory
        if self.scaler is None:
            self.optimizer['G'].zero_grad()
        else:
            with autocast(enabled=False):
                self.optimizer['G'].zero_grad()
        self.backward_G()
        if self.scaler is None:
            self.optimizer['G'].step()
        else:
            with autocast(enabled=False):
                self.scaler.step(self.optimizer['G'])

        # D_A and D_B
        self.set_requires_grad([self.model['D_A'], self.model['D_B']], True)
        if self.scaler is None:
            self.optimizer['D'].zero_grad()
        else:
            with autocast(enabled=False):
                self.optimizer['D'].zero_grad()
        self.backward_D()
        if self.scaler is None:
            self.optimizer['D'].step()
        else:
            with autocast(enabled=False):
                self.scaler.step(self.optimizer['D'])

        # save translated images
        if self._rank == 0:
            self.save_imgs(['real_A', 'real_B', 'fake_A', 'fake_B', 'rec_A', 'rec_B'])

        return 0

    def backward_G(self, retain_graph=False):
        """Calculate the loss for generators G_A and G_B"""
        # Adversarial loss D_A(G_A(B))
        loss_G_A = self.criterions['gan_G'](self.model['D_A'](self.fake_A), True)
        # Adversarial loss D_B(G_B(A))
        loss_G_B = self.criterions['gan_G'](self.model['D_B'](self.fake_B), True)
        loss_G = loss_G_A + loss_G_B

        # Forward cycle loss || G_A(G_B(A)) - A||
        loss_recon_A = self.criterions['recon'](self.rec_A, self.real_A)
        # Backward cycle loss || G_B(G_A(B)) - B||
        loss_recon_B = self.criterions['recon'](self.rec_B, self.real_B)
        loss_recon = loss_recon_A + loss_recon_B

        # G_A should be identity if real_B is fed: ||G_B(B) - B||
        idt_A = self.model['G_A'](self.real_B)
        loss_idt_A = self.criterions['ide'](idt_A, self.real_B)
        # G_B should be identity if real_A is fed: ||G_A(A) - A||
        idt_B = self.model['G_B'](self.real_A)
        loss_idt_B = self.criterions['ide'](idt_B, self.real_A)
        loss_idt = loss_idt_A + loss_idt_B

        # combined loss and calculate gradients
        loss = loss_G * self.cfg.TRAIN.LOSS.losses['gan_G'] + \
                loss_recon * self.cfg.TRAIN.LOSS.losses['recon'] + \
                loss_idt * self.cfg.TRAIN.LOSS.losses['ide']
        if self.scaler is None:
            loss.backward(retain_graph=retain_graph)
        else:
            with autocast(enabled=False):
                self.scaler.scale(loss).backward(retain_graph=retain_graph)

        meters = {'gan_G': loss_G.item(),
                  'recon': loss_recon.item(),
                  'ide': loss_idt.item()}
        self.train_progress.update(meters)

    def backward_D_basic(self, netD, real, fake, fake_pool):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterions['gan_D'](pred_real, True)
        # Fake
        pred_fake = netD(fake_pool.query(fake))
        loss_D_fake = self.criterions['gan_D'](pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss = loss_D * self.cfg.TRAIN.LOSS.losses['gan_D']
        if self.scaler is None:
            loss.backward()
        else:
            with autocast(enabled=False):
                self.scaler.scale(loss).backward()
        return loss_D.item()

    def backward_D(self):
        loss_D_A = self.backward_D_basic(self.model['D_A'], self.real_A, self.fake_A, self.fake_A_pool)
        loss_D_B = self.backward_D_basic(self.model['D_B'], self.real_B, self.fake_B, self.fake_B_pool)
        meters = {'gan_D': loss_D_A + loss_D_B}
        self.train_progress.update(meters)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_imgs(self, names):
        for name in names:
            img = getattr(self, name)[0]
            img_np = tensor2im(img, mean=self.cfg.DATA.norm_mean, std=self.cfg.DATA.norm_std)
            save_image(img_np, osp.join(self.save_dir, 'epoch_{}_{}.jpg'.format(self._epoch, name)))

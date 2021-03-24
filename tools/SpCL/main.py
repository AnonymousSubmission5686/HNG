import argparse
import collections
import shutil
import sys
import os
import os.path as osp
import time
from datetime import timedelta
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

from openunreid.apis import BaseRunner, batch_processor, test_reid, set_random_seed
from openunreid.core.solvers import build_lr_scheduler, build_optimizer
from openunreid.data import (
    build_test_dataloader,
    build_train_dataloader,
    build_val_dataloader,
)
from openunreid.models import build_model
from openunreid.models.losses import build_loss
from openunreid.models.utils.extract import extract_features
from openunreid.utils.config import (
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    log_config_to_file,
)
from openunreid.utils.dist_utils import init_dist, synchronize
from openunreid.utils.file_utils import mkdir_if_missing
from openunreid.utils.logger import Logger
from openunreid.utils.torch_utils import tensor2im
from openunreid.data.utils.data_utils import save_image

class SpCLRunner(BaseRunner):
    def save_imgs(self, input):
        for name,imgs in input.items():
            for i,img in enumerate(imgs):
                # img = img[0]
                img_np = tensor2im(img, mean=self.cfg.DATA.norm_mean, std=self.cfg.DATA.norm_std)
                save_image(img_np, osp.join(self.save_dir, 'epoch_{}_{}_{}.jpg'.format(self._epoch, name, i)))

    def update_labels(self):
        sep = "*************************"
        print(f"\n{sep} Start updating pseudo labels on epoch {self._epoch} {sep}\n")

        memory_features = []
        start_ind = 0
        for idx, dataset in enumerate(self.train_sets):
            if idx in self.cfg.TRAIN.unsup_dataset_indexes:
                memory_features.append(
                    self.criterions["hybrid_memory"]
                    .features[start_ind : start_ind + len(dataset)]
                    .clone()
                    .cpu()
                )
                start_ind += len(dataset)
            else:
                source_classes = dataset.num_pids
                start_ind += dataset.num_pids

        # generate pseudo labels
        pseudo_labels, label_centers = self.label_generator(
            self._epoch, memory_features=memory_features, print_freq=self.print_freq
        )

        self.model['id'].module.num_classes = label_centers[0].size(0)
        self.model['id'].module.add_module('classifier', nn.Linear(self.model['id'].module.num_features, self.model['id'].module.num_classes, bias=False))
        self.model['id'].module.classifier.cuda()

        # update train loader
        self.train_loader, self.train_sets = build_train_dataloader(
            self.cfg, pseudo_labels, self.train_sets, self._epoch, joint=False
        )

        # update criterions
        if "cross_entropy" in self.criterions.keys():
            self.criterions[
                "cross_entropy"
            ].num_classes = self.train_loader[0].loader.dataset.num_pids

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
        
        #if self.epoch == 0:
        #    self.optimizer['id'].add_param_group({'params': self.model['id'].module.classifier.weight})
        #else:
        #    self.optimizer['id'].param_groups[-1]['params'] = self.model['id'].module.classifier.weight

        # update memory labels
        memory_labels = []
        start_pid = 0
        for idx, dataset in enumerate(self.train_sets):
            if idx in self.cfg.TRAIN.unsup_dataset_indexes:
                labels = pseudo_labels[self.cfg.TRAIN.unsup_dataset_indexes.index(idx)]
                memory_labels.append(torch.LongTensor(labels) + start_pid)
                start_pid += max(labels) + 1
            else:
                num_pids = dataset.num_pids
                memory_labels.append(torch.arange(start_pid, start_pid + num_pids))
                start_pid += num_pids
        memory_labels = torch.cat(memory_labels).view(-1)
        self.criterions["hybrid_memory"]._update_label(memory_labels)

        print(f"\n{sep} Finished updating pseudo label {sep}\n")

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def train_step(self, iter, batch):
        start_ind, start_pid = 0, 0
        for idx, sub_batch in enumerate(batch):
            if idx in self.cfg.TRAIN.unsup_dataset_indexes:
                sub_batch["ind"] += start_ind
                start_ind += len(self.train_sets[idx])
            else:
                sub_batch["ind"] = sub_batch["id"] + start_ind
                start_ind += self.train_sets[idx].num_pids

            sub_batch["id"] += start_pid
            start_pid += self.train_sets[idx].num_pids
        data = batch_processor(batch, self.cfg.MODEL.dsbn)

        if isinstance(data["img"],list):
            x_ori = data["img"][0].cuda()
            x_aug = data["img"][1].cuda()
        else:
            inputs = data["img"].cuda()
        targets = data["id"].cuda()
        indexes = data["ind"].cuda()

        # x_ori_view = x_ori.view((x_ori.size(0), -1))
        # ori_norm = F.normalize(x_ori_view, p=2, dim=1)
        # sim_ori = ori_norm.mm(ori_norm.t())
        # _, idx_sort = sim_ori.sort(dim=1, descending=True)
        # neg_idx = idx_sort[:,4]
        # x_neg = x_ori_view[neg_idx].view((x_ori.size(0),x_ori.size(1),x_ori.size(2)))
        # targets_neg = targets[neg_idx]

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
            neg_target = random.choice(list(neg_targets))
            #neg_index = dic_target[neg_target]
            neg_index = np.random.choice(dic_target[neg_target])
            x_neg.append(x_ori[neg_index])
            targets_neg.append(neg_target)
        x_neg = torch.stack(x_neg)
        targets_neg = torch.tensor(targets_neg)

        x_gen_ori, x_gen_ori_neg = self.model['generator'](x_ori, x_neg, self.mix_weight)

        if self.cfg.TRAIN.stage == 2:
            # -------------------------------------------
            self.set_requires_grad([self.model['discriminator']], False)
            self.optimizer['generator'].zero_grad()
            loss_recon = self.criterions['recon'](x_gen_ori, x_ori) * 10
            loss_G = loss_recon
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

            spcl_loss, ad_loss, co_loss = self.criterions["hybrid_memory"](results, indexes)
            loss_id = spcl_loss+co_loss
            loss_all =  spcl_loss + ad_loss +co_loss

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
            self.save_imgs({'x_ori': x_ori, 'x_gen_ori_neg': x_gen_ori_neg, 'x_gen_ori': x_gen_ori})

        if self.cfg.TRAIN.stage == 2:
            return loss_D,loss_recon
        elif self.cfg.TRAIN.stage == 3:
            return spcl_loss,co_loss,ad_loss,loss_D,loss_recon,ce_loss


def parge_config():
    parser = argparse.ArgumentParser(description="SpCL training")
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--work-dir", help="the dir to save logs and models", default=""
    )
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["none", "pytorch", "slurm"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--tcp-port", type=str, default="5017")
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
        help="set extra config keys if needed",
    )
    args = parser.parse_args()

    cfg_from_yaml_file(args.config, cfg)
    assert cfg.TRAIN.PSEUDO_LABELS.use_outliers
    cfg.launcher = args.launcher
    cfg.tcp_port = args.tcp_port
    if not args.work_dir:
        args.work_dir = Path(args.config).stem
    cfg.work_dir = cfg.LOGS_ROOT / args.work_dir
    mkdir_if_missing(cfg.work_dir)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    shutil.copy(args.config, cfg.work_dir / "config.yaml")

    return args, cfg


def main():
    start_time = time.monotonic()

    # init distributed training
    args, cfg = parge_config()
    dist = init_dist(cfg)
    set_random_seed(cfg.TRAIN.seed, cfg.TRAIN.deterministic)
    synchronize()

    # init logging file
    logger = Logger(cfg.work_dir / "log.txt", debug=False)
    sys.stdout = logger
    print("==========\nArgs:{}\n==========".format(args))
    log_config_to_file(cfg)

    # build train loader
    train_loader, train_sets = build_train_dataloader(cfg, joint=False)

    # build model
    model = build_model(cfg, 0, init=cfg.MODEL.source_pretrained, init_G=cfg.MODEL.G_pretrained, init_D=cfg.MODEL.D_pretrained )
    for key in model.keys():
        model[key].cuda()
    # model.cuda()

    if dist:
        ddp_cfg = {
            "device_ids": [cfg.gpu],
            "output_device": cfg.gpu,
            "find_unused_parameters": True,
        }
        for key in model.keys():
            model[key] = DistributedDataParallel(model[key], **ddp_cfg)
        # model = DistributedDataParallel(model, **ddp_cfg)
    elif cfg.total_gpus > 1:
        # model = DataParallel(model)
        for key in model.keys():
            model[key] = DataParallel(model[key])

    # build optimizer
    optimizer = {}

    optimizer['id'] = build_optimizer([model['id']], **cfg.TRAIN.OPTIM2)
    optimizer['generator'] = build_optimizer([model['generator']], **cfg.TRAIN.OPTIM)
    optimizer['discriminator'] = build_optimizer([model['discriminator']], **cfg.TRAIN.OPTIM)

    # build lr_scheduler
    if cfg.TRAIN.SCHEDULER.lr_scheduler is not None:
        # lr_scheduler = build_lr_scheduler(optimizer, **cfg.TRAIN.SCHEDULER)
        lr_scheduler = [build_lr_scheduler(optimizer[key], **cfg.TRAIN.SCHEDULER) \
                        for key in optimizer.keys()]
    else:
        lr_scheduler = None

    # build loss functions
    num_memory = 0
    for idx, set in enumerate(train_sets):
        if idx in cfg.TRAIN.unsup_dataset_indexes:
            # instance-level memory for unlabeled data
            num_memory += len(set)
        else:
            # class-level memory for labeled data
            num_memory += set.num_pids

    if isinstance(model, (DataParallel, DistributedDataParallel)):
        num_features = model.module.num_features
    else:
        num_features = model['id'].module.num_features

    criterions = build_loss(
        cfg.TRAIN.LOSS,
        num_classes=0,
        num_features=num_features,
        num_memory=num_memory,
        cuda=True,
    )

    # init memory
    loaders, datasets = build_val_dataloader(
        cfg, for_clustering=True, all_datasets=True
    )
    memory_features = []
    for idx, (loader, dataset) in enumerate(zip(loaders, datasets)):
        features = extract_features(
            model['id'], loader, dataset, with_path=False, prefix="Extract: ",
        )
        assert features.size(0) == len(dataset)
        if idx in cfg.TRAIN.unsup_dataset_indexes:
            # init memory for unlabeled data with instance features
            memory_features.append(features)
        else:
            # init memory for labeled data with class centers
            centers_dict = collections.defaultdict(list)
            for i, (_, pid, _) in enumerate(dataset):
                centers_dict[pid].append(features[i].unsqueeze(0))
            centers = [
                torch.cat(centers_dict[pid], 0).mean(0)
                for pid in sorted(centers_dict.keys())
            ]
            memory_features.append(torch.stack(centers, 0))
    del loaders, datasets

    memory_features = torch.cat(memory_features)
    criterions["hybrid_memory"]._update_feature(memory_features)

    # build runner
    runner = SpCLRunner(
        cfg,
        model,
        optimizer,
        criterions,
        train_loader,
        train_sets=train_sets,
        lr_scheduler=lr_scheduler,
        meter_formats={"Time": ":.3f",},
        reset_optim=False,
    )

    # resume
    if args.resume_from:
        runner.resume(args.resume_from)

    # start training
    runner.run()

    # load the best model
    runner.resume(cfg.work_dir / "model_best_id.pth")

    # final testing
    test_loaders, queries, galleries = build_test_dataloader(cfg)
    for i, (loader, query, gallery) in enumerate(zip(test_loaders, queries, galleries)):
        cmc, mAP = test_reid(
            cfg, model['id'], loader, query, gallery, dataset_name=cfg.TEST.datasets[i]
        )

    # print time
    end_time = time.monotonic()
    print("Total running time: ", timedelta(seconds=end_time - start_time))


if __name__ == "__main__":
    main()

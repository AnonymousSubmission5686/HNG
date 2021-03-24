# Written by xxx
import os
import copy
import warnings
import random

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from ..utils.dist_utils import convert_sync_bn, get_dist_info, simple_group_split
from ..utils.torch_utils import copy_state_dict, load_checkpoint
from .backbones import build_bakcbone
from .layers import build_embedding_layer, build_pooling_layer
from .utils.dsbn_utils import convert_dsbn

__all__ = ["ReIDBaseModel", "TeacherStudentNetwork",
            "build_model", "build_gan_model"]


class ReIDBaseModel(nn.Module):
    """
    Base model for object re-ID, which contains
    + one backbone, e.g. ResNet50
    + one global pooling layer, e.g. avg pooling
    + one embedding block, (linear, bn, relu, dropout) or (bn, dropout)
    + one classifier
    """

    def __init__(
        self,
        arch,
        num_classes,
        pooling="avg",
        embed_feat=0,
        dropout=0.0,
        num_parts=0,
        include_global=True,
        pretrained=True,
    ):
        super(ReIDBaseModel, self).__init__()
        # model = {}

        self.backbone = build_bakcbone(arch, pretrained=pretrained)

        self.global_pooling = build_pooling_layer(pooling)
        self.head = build_embedding_layer(
            self.backbone.num_features, embed_feat, dropout
        )
        # model['generator'] = self.generator
        # model['backbone'] = nn.Sequential(self.backbone,self.global_pooling,self.head)
        self.num_features = self.head.num_features

        self.num_classes = num_classes
        if self.num_classes > 0:
            self.classifier = nn.Linear(self.head.num_features, num_classes, bias=False)
            init.normal_(self.classifier.weight, std=0.001)

        self.num_parts = num_parts
        self.include_global = include_global

        if not pretrained:
            self.reset_params()

    @torch.no_grad()
    def initialize_centers(self, centers, labels):
        if self.num_classes > 0:
            # if use_all:
            #     self.classifier.weight.data.copy_(centers.to(self.classifier.weight.device))
            # else:
            self.classifier.weight.data[
                labels.min().item() : labels.max().item() + 1
            ].copy_(centers.to(self.classifier.weight.device))
        else:
            warnings.warn(
                f"there is no classifier in the {self.__class__.__name__}, "
                f"the initialization does not function"
            )

    def forward(self, x, val=True):
        if isinstance(x,list):
            batch_size = x[0].size(0)
            x_ori = x[0]
            x_aug = x[1]
            memory = x[2]
            targets = x[3]
        else:
            x_ori = x
            batch_size = x.size(0)
        results = {}


        backbone_ori = self.backbone(x_ori)

        out = self.global_pooling(backbone_ori)
        out = out.view(batch_size, -1)

        if self.num_parts > 1:
            assert x.size(2) % self.num_parts == 0
            x_split = torch.split(x, x.size(2) // self.num_parts, dim=2)
            outs = []
            if self.include_global:
                outs.append(out)
            for subx in x_split:
                outs.append(self.global_pooling(subx).view(batch_size, -1))
            out = outs

        results["pooling"] = out

        if isinstance(out, list):
            feat = [self.head(f) for f in out]
        else:
            feat = self.head(out)

        results["feat"] = feat

        if not self.training and val:
            return feat

        if self.num_classes > 0:
            if isinstance(feat, list):
                prob = [self.classifier(f) for f in feat]
            else:
                prob = self.classifier(feat)

            results["prob"] = prob

        # if self.training:
        #     index = list(range(len(targets)))
        #     dic_taget = dict(zip(targets.tolist(), index))
        #     x_neg = []
        #     for i, target in enumerate(targets):
        #         neg_targets = set(targets.tolist())-{target.item()}
        #         neg_index = dic_taget[random.choice(list(neg_targets))]
        #         x_neg.append(x_ori[neg_index])
        #     x_neg = torch.stack(x_neg)
        #     backbone_neg = self.backbone(x_neg)
        #     # labels = memory.labels.clone()
        #     # cluster_sum = torch.zeros(labels.max() + 1, 2048).float()
        #     # cluster_sum.index_add_(0, labels.cpu(), memory.features.clone().cpu())
        #     # nums = torch.zeros(labels.max() + 1, 1).float()
        #     # nums.index_add_(0, labels.cpu(), torch.ones(memory.num_memory, 1).float().cpu())
        #     # centroid = cluster_sum/nums
        #     #
        #     # norm_feat = F.normalize(feat, p=2, dim=1).cpu()
        #     # neg_centroid_idx = norm_feat.mm(centroid.t()).sort().indices[:, -100]
        #     # neg_centroid = centroid[neg_centroid_idx]
        #     # gen_input_ori_neg = torch.cat([backbone_ori, backbone_neg], 1)
        #     gen_input_ori_neg = torch.add(0.5*backbone_ori, 0.5*backbone_neg)
        #     x_gen_ori_neg = self.generator(gen_input_ori_neg)    #4096 x 16 x 8
        #
        #     # gen_input_neg_ori = torch.cat([backbone_neg, backbone_ori], 1)
        #     # x_gen_neg_ori = self.generator(gen_input_neg_ori)  # 4096 x 16 x 8
        #
        #     gen_input_ori = torch.add(0.5*backbone_ori, 0.5*backbone_ori)
        #     # gen_input_ori = torch.cat([backbone_ori, backbone_ori], 1)
        #     x_gen_ori = self.generator(gen_input_ori)
        #
        #     # backbone_gen = self.backbone(x_gen_ori_neg)
        #     # out_gen = self.global_pooling(backbone_gen)
        #     # out_gen = out_gen.view(batch_size, -1)
        #     # results["pooling_gen"] = out_gen
        #     # if isinstance(out_gen, list):
        #     #     feat_gen = [self.head(f) for f in out_gen]
        #     # else:
        #     #     feat_gen = self.head(out_gen)
        #     # results["feat_gen"] = feat_gen
        #     #
        #     # backvone_aug = self.backbone(x_aug)
        #     # out_aug = self.global_pooling(backvone_aug)
        #     # out_aug = out_aug.view(batch_size, -1)
        #     # results["pooling_aug"] = out_aug
        #     # if isinstance(out_aug, list):
        #     #     feat_aug = [self.head(f) for f in out_aug]
        #     # else:
        #     #     feat_aug = self.head(out_aug)
        #     # results["feat_aug"] = feat_aug
        #
        #     d_x_ori = self.discriminator(x_ori)
        #     d_x_gen_ori_neg = self.discriminator(x_gen_ori_neg.detach())
        #     # d_x_gen_neg_ori = self.discriminator(x_gen_neg_ori)
        #     d_x_gen_ori = self.discriminator(x_gen_ori.detach())

        # return results, x_ori, x_neg, x_gen_ori_neg, x_gen_ori, d_x_ori, d_x_gen_ori_neg, d_x_gen_ori
        return results, backbone_ori

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class TeacherStudentNetwork(nn.Module):
    """
    TeacherStudentNetwork.
    """

    def __init__(
        self, net, alpha=0.999,
    ):
        super(TeacherStudentNetwork, self).__init__()
        self.net = net
        self.mean_net = copy.deepcopy(self.net)

        for param, param_m in zip(self.net.parameters(), self.mean_net.parameters()):
            param_m.data.copy_(param.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        self.alpha = alpha

    def forward(self, x):
        if not self.training:
            return self.mean_net(x)

        results = self.net(x)

        with torch.no_grad():
            self._update_mean_net()  # update mean net
            results_m = self.mean_net(x)

        return results, results_m

    @torch.no_grad()
    def initialize_centers(self, centers, labels):
        self.net.initialize_centers(centers, labels)
        self.mean_net.initialize_centers(centers, labels)

    @torch.no_grad()
    def _update_mean_net(self):
        for param, param_m in zip(self.net.parameters(), self.mean_net.parameters()):
            param_m.data.mul_(self.alpha).add_(param.data, alpha=1-self.alpha)


def build_model(
    cfg, num_classes, init=None, init_G=None, init_D=None
):
    """
    Build a (cross-domain) re-ID model
    with domain-specfic BNs (optional)
    """
    model = {}
    model['generator'] = build_bakcbone(cfg.MODEL.generator)
    model['discriminator'] = build_bakcbone('patchgan_3layers')
    # construct a reid model

    model['id'] = ReIDBaseModel(
        cfg.MODEL.backbone,
        num_classes,
        cfg.MODEL.pooling,
        cfg.MODEL.embed_feat,
        cfg.MODEL.dropout,
        pretrained=cfg.MODEL.imagenet_pretrained,
    )

    # convert to domain-specific bn (optional)
    num_domains = len(cfg.TRAIN.datasets.keys())
    if (num_domains > 1) and cfg.MODEL.dsbn:
        if cfg.TRAIN.val_dataset in list(cfg.TRAIN.datasets.keys()):
            target_domain_idx = list(cfg.TRAIN.datasets.keys()).index(
                cfg.TRAIN.val_dataset
            )
        else:
            target_domain_idx = -1
            warnings.warn(
                f"the domain of {cfg.TRAIN.val_dataset} for validation is not within "
                f"train sets, we use "
                f"{list(cfg.TRAIN.datasets.keys())[-1]}'s BN intead, "
                f"which may cause unsatisfied performance."
            )
        convert_dsbn(model['id'], num_domains, target_domain_idx)
        convert_dsbn(model['generator'], num_domains, target_domain_idx)
        convert_dsbn(model['discriminator'], num_domains, target_domain_idx)
    else:
        if cfg.MODEL.dsbn:
            warnings.warn(
                "domain-specific BN is switched off, since there's only one domain."
            )
        cfg.MODEL.dsbn = False

    # load source-domain pretrain (optional)
    if init is not None:
        state_dict = load_checkpoint(init)
        if "state_dict_1" in state_dict.keys():
            state_dict = state_dict["state_dict_1"]
        else:
            state_dict = state_dict["state_dict"]
        copy_state_dict(state_dict, model['id'], strip="module.")

    if init_G is not None:
        state_dict = load_checkpoint(init_G)
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        copy_state_dict(state_dict, model['generator'], strip="module.")

    if init_D is not None:
        state_dict = load_checkpoint(init_D)
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        copy_state_dict(state_dict, model['discriminator'], strip="module.")


    # create mean network (optional)
    if cfg.MODEL.mean_net:
        model = TeacherStudentNetwork(model, cfg.MODEL.alpha)

    # convert to sync bn (optional)
    rank, world_size, dist = get_dist_info()
    if cfg.MODEL.sync_bn and dist:
        if cfg.TRAIN.LOADER.samples_per_gpu < cfg.MODEL.samples_per_bn:

            total_batch_size = cfg.TRAIN.LOADER.samples_per_gpu * world_size
            if total_batch_size > cfg.MODEL.samples_per_bn:
                assert (
                    total_batch_size % cfg.MODEL.samples_per_bn == 0
                ), "Samples for sync_bn cannot be evenly divided."
                group_num = int(total_batch_size // cfg.MODEL.samples_per_bn)
                dist_groups = simple_group_split(world_size, rank, group_num)
            else:
                dist_groups = None
                warnings.warn(
                    f"'Dist_group' is switched off, since samples_per_bn "
                    f"({cfg.MODEL.samples_per_bn,}) is larger than or equal to "
                    f"total_batch_size ({total_batch_size})."
                )
            convert_sync_bn(model, dist_groups)

        else:
            warnings.warn(
                f"Sync BN is switched off, since samples ({cfg.MODEL.samples_per_bn, })"
                f" per BN are fewer than or same as samples "
                f"({cfg.TRAIN.LOADER.samples_per_gpu}) per GPU."
            )
            cfg.MODEL.sync_bn = False

    else:
        if cfg.MODEL.sync_bn and not dist:
            warnings.warn(
                "Sync BN is switched off, since the program is running without DDP"
            )
        cfg.MODEL.sync_bn = False

    return model


def build_gan_model(
    cfg,
    only_generator = False,
):
    """
    Build a domain-translation model
    """
    model = {}

    if only_generator:
        # for inference
        model['G'] = build_bakcbone(cfg.MODEL.generator)
    else:
        # construct generators
        model['G_A'] = build_bakcbone(cfg.MODEL.generator)
        model['G_B'] = build_bakcbone(cfg.MODEL.generator)
        # construct discriminators
        model['D_A'] = build_bakcbone(cfg.MODEL.discriminator)
        model['D_B'] = build_bakcbone(cfg.MODEL.discriminator)
        # construct a metric net for spgan
        if cfg.MODEL.spgan:
            model['Metric'] = build_bakcbone('metricnet')

    # convert to sync bn (optional)
    rank, world_size, dist = get_dist_info()
    if cfg.MODEL.sync_bn and dist:
        if cfg.TRAIN.LOADER.samples_per_gpu < cfg.MODEL.samples_per_bn:

            total_batch_size = cfg.TRAIN.LOADER.samples_per_gpu * world_size
            if total_batch_size > cfg.MODEL.samples_per_bn:
                assert (
                    total_batch_size % cfg.MODEL.samples_per_bn == 0
                ), "Samples for sync_bn cannot be evenly divided."
                group_num = int(total_batch_size // cfg.MODEL.samples_per_bn)
                dist_groups = simple_group_split(world_size, rank, group_num)
            else:
                dist_groups = None
                warnings.warn(
                    f"'Dist_group' is switched off, since samples_per_bn "
                    f"({cfg.MODEL.samples_per_bn,}) is larger than or equal to "
                    f"total_batch_size ({total_batch_size})."
                )

            for key in model.keys():
                convert_sync_bn(model[key], dist_groups)

        else:
            warnings.warn(
                f"Sync BN is switched off, since samples ({cfg.MODEL.samples_per_bn, })"
                f" per BN are fewer than or same as samples "
                f"({cfg.TRAIN.LOADER.samples_per_gpu}) per GPU."
            )
            cfg.MODEL.sync_bn = False

    else:
        if cfg.MODEL.sync_bn and not dist:
            warnings.warn(
                "Sync BN is switched off, since the program is running without DDP"
            )
        cfg.MODEL.sync_bn = False

    return model

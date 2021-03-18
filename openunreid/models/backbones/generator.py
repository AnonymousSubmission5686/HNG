# Modified from https://github.com/Simon4Yan/eSPGAN/blob/master/py-spgan/models/models_spgan.py

import functools
import torch.nn as nn
import torch
import torch.nn.functional as F

from ..utils.init_net import init_weights


__all__ = ['ResnetGenerator', 'resnet_6blocks', 'resnet_9blocks']


def conv_norm_relu(in_dim, out_dim, kernel_size, stride, padding=0,
                  norm = functools.partial(nn.InstanceNorm2d, affine=False), relu=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        norm(out_dim),
        relu())


def dconv_norm_relu(in_dim, out_dim, kernel_size, stride, padding=0,
                   output_padding=0, norm = functools.partial(nn.InstanceNorm2d, affine=False), relu=nn.ReLU):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride,
                           padding, output_padding, bias=False),
        norm(out_dim),
        relu())


class ResiduleBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResiduleBlock, self).__init__()

        self.layers = nn.Sequential(nn.ReflectionPad2d(1),
                                conv_norm_relu(in_dim, out_dim, 3, 1),
                                nn.ReflectionPad2d(1),
                                nn.Conv2d(out_dim, out_dim, 3, 1),
                                nn.InstanceNorm2d(out_dim))
    def forward(self, x):
        return x + self.layers(x)


class ResnetGenerator(nn.Module):
    '''
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    '''

    def __init__(self, n_blocks, dim=64):
        super(ResnetGenerator, self).__init__()

        layers_en = [nn.ReflectionPad2d(3),
                    conv_norm_relu(3, dim * 1, 7, 1),
                    conv_norm_relu(dim * 1, dim * 2, 3, 2, 1),
                    conv_norm_relu(dim * 2, dim * 4, 3, 2, 1)]    # out: 256 x 64 x 32
        # layers = [
        #             dconv_norm_relu(2048, 1024, 3, 2, 1, 1),
        #             dconv_norm_relu(1024, 256, 3, 2, 1, 1),
        #          ]
        for _ in range(int(n_blocks/2)):
            layers_en += [ResiduleBlock(dim * 4, dim * 4)]
        self.encoder = nn.Sequential(*layers_en)

        for _ in range(int(n_blocks/2)):
            layers_de = [ResiduleBlock(dim * 4, dim * 4)]

        # layers_de = [nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
        #              ResiduleBlock(dim * 4, dim * 4),
        #              ResiduleBlock(dim * 4, dim * 4),
        #              ResiduleBlock(dim * 4, dim * 4)]

        layers_de += [dconv_norm_relu(dim * 4, dim * 2, 3, 2, 1, 1),
                    dconv_norm_relu(dim * 2, dim * 1, 3, 2, 1, 1),
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(dim, 3, 7, 1),
                    nn.Tanh()]
        self.decoder = nn.Sequential(*layers_de)

        init_weights(self.encoder)
        init_weights(self.decoder)

    def forward(self, x_ori, x_neg, mix_weight):
        x_ori_en = self.encoder(x_ori)
        x_neg_en = self.encoder(x_neg)

        #x_ori_en_view = x_ori_en.view((x_ori_en.size(0), -1))
        #ori_norm = F.normalize(x_ori_en_view, p=2, dim=1)
        #sim_ori = ori_norm.mm(ori_norm.t())
        #_, idx_sort = sim_ori.sort(dim=1, descending=True)
        #neg_idx = idx_sort[:, 15]
        #x_neg_en = x_ori_en_view[neg_idx].view(x_ori_en.size())
        #targets_neg = targets[neg_idx]

        de_input_ori = torch.add(0.5 * x_ori_en, 0.5 * x_ori_en)
        de_input_ori_neg = torch.add(mix_weight * x_ori_en, (1-mix_weight) * x_neg_en)

        # de_input_ori = torch.cat((x_ori_en, x_ori_en), 1)
        # de_input_ori_neg = torch.cat((x_ori_en, x_neg_en), 1)

        x_gen_ori = self.decoder(de_input_ori)
        x_gen_ori_neg = self.decoder(de_input_ori_neg)

        return x_gen_ori, x_gen_ori_neg

def resnet_9blocks(pretrained=False, **kwargs):
    r"""Generator with 9 residual blocks
    """
    return ResnetGenerator(
        9, **kwargs
    )

def resnet_6blocks(pretrained=False, **kwargs):
    r"""Generator with 6 residual blocks
    """
    return ResnetGenerator(
        6, **kwargs
    )

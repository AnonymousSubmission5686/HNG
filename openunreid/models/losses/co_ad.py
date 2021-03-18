import torch
import torch.nn as nn
import torch.nn.functional as F

class co_ad(nn.Module):
    def __init__(self):
        super(co_ad, self).__init__()
        self.temp = 0.05

    def forward(self, results, targets):
        ori = results["feat"]
        ori_norm = F.normalize(ori, p=2, dim=1)
        sim_ori = ori_norm.mm(ori_norm.t()) / self.temp

        ########### instance contrast ################
        gen = results["feat_gen"]
        gen_norm = F.normalize(gen, p=2, dim=1)
        sim_gen = ori_norm.mm(gen_norm.t()) / self.temp

        aug = results["feat_aug"]
        aug_norm = F.normalize(aug, p=2, dim=1)
        sim_aug = ori_norm.mm(aug_norm.t()) / self.temp

        exp_ori = torch.exp(sim_ori)
        exp_gen = torch.exp(sim_gen)
        exp_aug = torch.exp(sim_aug)
        B = ori.size(0)
        # exp_ori = exp_ori*(torch.ones(B,B)-torch.eye(B)).cuda()
        co_loss = 0
        for i, (exp_o, exp_a, exp_g, target) in enumerate(zip(exp_ori, exp_aug, exp_gen, targets)):
            exp_sum = 0
            exp_same = []
            for o, a, t in zip(exp_o, exp_a, targets):
                if t == target:
                    exp_same.append(o)
                    exp_same.append(a)
                if not t == target:
                    exp_sum = exp_sum + (o + a)
            l = 0
            for e in exp_same:
                l = l + (-torch.log((e / (e + exp_sum + exp_g.sum() + 1e-6)) + 1e-6))
            co_loss = co_loss + l
        co_loss = co_loss / B

        ad_loss = 0
        for i, (exp_o, exp_a, exp_g, target) in enumerate(zip(exp_ori, exp_aug, exp_gen, targets)):
            exp_sum = 0
            exp_same = []
            for g, t in zip(exp_g, targets):
                if t == target:
                    exp_same.append(g)
                if not t == target:
                    exp_sum = exp_sum + g
            l = 0
            for e in exp_same:
                l = l + (-torch.log((e / (e + exp_sum + exp_a.sum() + exp_o.sum() + 1e-6)) + 1e-6))
            ad_loss = ad_loss + l
        ad_loss = ad_loss / B

        return ad_loss, co_loss
# Ge et al. Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID.  # noqa
# Written by xxx.

import torch
import torch.nn.functional as F
from torch import autograd, nn

torch.autograd.set_detect_anomaly(True)
from ...utils.dist_utils import all_gather_tensor
import pdb

try:
    # PyTorch >= 1.6 supports mixed precision training
    from torch.cuda.amp import custom_fwd, custom_bwd


    class HM(autograd.Function):

        @staticmethod
        @custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, inputs, indexes, features, momentum):
            ctx.features = features
            ctx.momentum = momentum
            outputs = inputs.mm(ctx.features.t())
            all_inputs = all_gather_tensor(inputs)
            all_indexes = all_gather_tensor(indexes)
            ctx.save_for_backward(all_inputs, all_indexes)
            return outputs

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_outputs):
            inputs, indexes = ctx.saved_tensors
            grad_inputs = None
            if ctx.needs_input_grad[0]:
                grad_inputs = grad_outputs.mm(ctx.features)

            # momentum update
            for x, y in zip(inputs, indexes):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

            return grad_inputs, None, None, None
except:
    class HM(autograd.Function):

        @staticmethod
        def forward(ctx, inputs, indexes, features, momentum):
            ctx.features = features
            ctx.momentum = momentum
            outputs = inputs.mm(ctx.features.t())
            all_inputs = all_gather_tensor(inputs)
            all_indexes = all_gather_tensor(indexes)
            ctx.save_for_backward(all_inputs, all_indexes)
            return outputs

        @staticmethod
        def backward(ctx, grad_outputs):
            inputs, indexes = ctx.saved_tensors
            grad_inputs = None
            if ctx.needs_input_grad[0]:
                grad_inputs = grad_outputs.mm(ctx.features)

            # momentum update
            for x, y in zip(inputs, indexes):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

            return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(
        inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device)
    )


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_memory, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_memory = num_memory

        self.momentum = momentum
        self.temp = temp

        self.register_buffer("features", torch.zeros(num_memory, num_features))
        self.register_buffer("labels", torch.zeros(num_memory).long())

    @torch.no_grad()
    def _update_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.features.data.copy_(features.float().to(self.features.device))

    @torch.no_grad()
    def _update_label(self, labels):
        self.labels.data.copy_(labels.long().to(self.labels.device))

    def forward(self, results, indexes):
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

        # inputs: B*2048, features: L*2048
        inputs = hm(aug_norm, indexes, self.features, self.momentum)
        inputs /= self.temp
        B = ori.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return masked_exps / masked_sums, masked_sums

        targets = self.labels[indexes].clone()
        labels = self.labels.clone()

        sim = torch.zeros(labels.max() + 1, B).float().cuda()
        sim.index_add_(0, labels, inputs.t().contiguous())
        nums = torch.zeros(labels.max() + 1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_memory, 1).float().cuda())
        mask = (nums > 0).float()
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)
        masked_sim, masked_sums = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
        spcl_loss = F.nll_loss(torch.log(masked_sim + 1e-6), targets)

        # exp_sim = torch.exp(sim.t())
        # exp_sim_sums = exp_sim.sum(1, keepdim=True).clone()- + 1e-6
        exp_ori = torch.exp(sim_ori)
        exp_gen = torch.exp(sim_gen)
        exp_aug = torch.exp(sim_aug)

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
                    exp_sum = exp_sum+(o + a)
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

        # ad_softmax = exp_gen / (masked_sums - exp_sim[:, targets].diag().unsqueeze(1) + exp_gen.sum(1, keepdim=True))
        # # pdb.set_trace()
        # ad_loss = F.nll_loss(torch.log(ad_softmax + 1e-6), torch.range(0, B - 1, dtype=torch.int64).cuda())

        # co_loss = torch.log(1 + (exp_gen.diag() / exp_aug.diag())).sum() / B

        # masked_sums_all = masked_sums + exp_gen.sum(1,keepdim=True) + exp_aug.sum(1,keepdim=True)
        # co_softmax = exp_aug / masked_sums_all
        # co_loss = F.nll_loss(torch.log(co_softmax + 1e-6), torch.range(0,B-1,dtype=torch.int64).cuda())

        return spcl_loss, ad_loss, co_loss

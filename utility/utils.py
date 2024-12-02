import numpy as np
import torch


def cond2weight(weight, sparsity=0.):
    cond_mean = 34.05538
    cond_std = 5.32269
    pos = torch.randn_like(weight) * cond_std + cond_mean
    neg = torch.randn_like(weight) * cond_std + cond_mean
    if sparsity != 0:
        mask = torch.ones_like(pos).flatten()
        idx = int(mask.numel() * sparsity)
        mask[: idx] = 0
        pos_idx = torch.randperm(mask.numel())
        neg_idx = torch.randperm(mask.numel())
        pos = pos * mask[pos_idx].view(pos.shape)
        neg = neg * mask[neg_idx].view(neg.shape)
    return pos, neg


def replace_model_weight(model: torch.nn.Module, sparsity=0.5):
    model_with_cond = {}
    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            # p = torch.nn.Parameter(cond2weight(param, sparsity))
            pos, neg = cond2weight(param, sparsity)
            param.data = pos - neg
            model_with_cond[name] = (pos, neg)
    return model, model_with_cond


def replace_model_weight_with_cond(model: torch.nn.Module, model_with_cond: dict, noise, seg=False):
    # get all names in model_with_cond 
    names = [name for name in model_with_cond.keys()]

    for name, param in model.named_parameters():
        if name in names:
            if ('conv2' in name) and seg:
                continue
            pos, neg = model_with_cond[name]
            pos, neg = pos.cuda(), neg.cuda()
            # add noise to pos and neg
            if noise:
                pos = pos + torch.randn_like(param) * noise * pos
                neg = neg + torch.randn_like(param) * noise * neg
            param.data = (pos - neg).cuda()

    return model


def replace_weight_absolute_normal(model: torch.nn.Module, model_with_cond: dict, noise):
    # get all names in model_with_cond 
    names = [name for name in model_with_cond.keys()]

    for name, param in model.named_parameters():
        if name in names:
            pos, neg = model_with_cond[name]
            pos, neg = pos.cuda(), neg.cuda()
            # add noise to pos and neg
            if noise:
                # pos = pos + torch.randn_like(param) * noise * pos
                # neg = neg + torch.randn_like(param) * noise * neg
                pos = pos + torch.randn_like(param) * noise
                neg = neg + torch.randn_like(param) * noise
            param.data = (pos - neg).cuda()

    return model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily
from time import time
import numpy as np
import os
import sys
sys.path.append('./models')
from noise_layers import NoiseModule, NoiseConv, NoiseConv1d


def model_weight_gen(model, w_type='mixture_normal', mean=1, std=0.01, sparsity=0.5, scaling=0.001, **kwargs):
    state_dict = {}
    for name, weight in model.named_parameters():
        # if ('weight' in name) and ('conv' in name):
        if 'conv' in name:
            weight = sparse_weight_gen(weight.shape, r_type=w_type, mean=mean, std=std, sparsity=sparsity, **kwargs)
            # weight should be non-negative for non-zero gaussian
            if w_type == 'normal' and mean != 0:
                weight = weight.abs()
                # weight = torch.where(weight < 0, torch.tensor(0).to(weight.dtype).to(weight.device), weight)
            weight = scaling * weight
            weight = nn.Parameter(weight)
        state_dict[name] = weight
    for i in range(3):
        state_dict[f'sa.{i}.mlp_bns.0.running_mean'] = model.sa[i].mlp_bns[0].running_mean
        state_dict[f'sa.{i}.mlp_bns.0.running_var'] = model.sa[i].mlp_bns[0].running_var
    model.load_state_dict(state_dict)
    return model


def sparse_weight_gen(shape, r_type='normal', mean=1, std=0.01, sparsity=0.5, device='cuda', **kwargs):
    if r_type == 'normal':
        weight = torch.randn(shape, device=device) * std + mean
        mask = torch.ones(weight.numel(), device=device)
        mask[:int(weight.numel() * sparsity)] = 0
        mask = mask[torch.randperm(mask.numel(), device=device)]
        weight = weight * mask.view(weight.shape)
        return weight
    elif r_type == 'mixture_normal':
        mix_ratio = kwargs['mix_ratio']
        mix = D.Categorical(torch.tensor(mix_ratio))
        comp = D.Normal(torch.tensor([-1, 0, 1]).to(float), torch.tensor([0.01, 0.01, 0.01]).to(float))
        gmm = MixtureSameFamily(mix, comp)
        weight = gmm.sample(shape)
        return weight.to(device)
    elif r_type == 'xavier':
        return torch.nn.init.xavier_normal_(torch.empty(shape, device=device))
    else:
        raise NotImplementedError


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def topK():
    pass


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def l1_distance(src, dst):
    """
    Calculate L1 distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """

    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = torch.cdist(src, dst, p=1)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint, distance_type='l2'):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # torch.manual_seed(1)

    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # calculating distance
        if distance_type == 'l2':
            dist = torch.sum((xyz - centroid) ** 2, -1)
        elif distance_type == 'l1':
            dist = torch.sum(torch.abs(xyz - centroid), -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids



def query_ball_point(radius, nsample, xyz, new_xyz, distance_type='l2'):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    if distance_type == 'l2':
        dist = square_distance(new_xyz, xyz)
    elif distance_type == 'l1':
        dist = l1_distance(new_xyz, xyz)
    group_idx[dist > radius ** 2] = N

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False, distance_type='l2'):
    """
    Input:
        npoint: number of centroids
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint, distance_type=distance_type) # [B, npoint, C]   seems to be [B, npoints]
    new_xyz = index_points(xyz, fps_idx)     # [B, npoint, C]
    idx = query_ball_point(radius, nsample, xyz, new_xyz, distance_type=distance_type)   # [B, npoint, nsample]  for each centroid, there are nsample neighbors
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class NewGraphSetAbstraction(NoiseModule):
    def __init__(self, npoint, radius,
                 nsample, in_channel, mlp,
                 group_all, noise=0, quantize='full',
                 distancing='l2', act=F.relu, hardweight=None,
                 mode=None, quant_bit=6):
        super(NewGraphSetAbstraction, self).__init__()
        self.npoint = npoint    # number of centroids
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        # scaling factors
        self.scaling = nn.Parameter(torch.ones(1) * 0.005)
        self.scaling.requires_grad = True

        last_channel = in_channel
        for out_channel in mlp[:2]:
            if quantize == 'full':
                self.mlp_convs.append(NoiseConv(last_channel, out_channel, 1,
                                                noise=noise, hard_weight=hardweight, mode=mode,
                                                quant=quant_bit))

            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

        self.act = act

        # self.fc = nn.Linear(mlp[-1], mlp[-1])

        self.noise = noise
        self.distance_type= distancing

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, distance_type=self.distance_type)

        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]


        bn = self.mlp_bns[0]
        conv = self.mlp_convs[0]
        new_points = bn(conv(new_points * self.scaling))
        # new_points = conv(new_points) * self.scaling
        # new_points = bn(conv(new_points))

        new_points = torch.max(new_points, 2)[0]
        # new_points = torch.mean(new_points, 2)[0]
        # new_points = F.relu(new_points)
        new_points = self.act(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class FeaturePropagation(NoiseModule):
    def __init__(self, in_channel,
                 mlp,
                 noise=0, quantize='full', 
                 hardweight=None,
                 mode=None, quant_bit=6):
        super(FeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(NoiseConv1d(last_channel, out_channel, 1,
                                              noise=noise, hard_weight=hardweight, mode=mode,
                                              quant=quant_bit))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        self.noise = noise
        self.scaling = nn.Parameter(torch.ones(1) * 0.005)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]  # N is the number of output points
            xyz2: sampled input points position data, [B, C, S] # S is the number of input centroids 
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N] 
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            '''point1中的每个点找到三个与其距离最近的点, 然后用这三个点进行插值'''
            dist_recip = 1.0 / (dists + 1e-8) # reciprocal of distance
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm      # normalize the receprocal of distance
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points * self.scaling)))
        return new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


def get_activation(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'square':
        return torch.square
    elif activation == 'mise':
        return nn.Mish()
    else:
        raise NotImplementedError

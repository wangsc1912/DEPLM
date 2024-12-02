import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import NewGraphSetAbstraction
from noise_layers import NoiseModule, NoiseLinear


class get_model(NoiseModule):
    def __init__(self, num_class,
                 normal_channel=True, c_prune_rate=1,
                 iter=[1,1,1], noise=0, quantize='full',
                 num_feat=1024, num_fc=1,
                 distancing='l2', act=F.relu,
                 radius=0.2, radius_multiple=2,
                 hardweight=None, scaling=False, hard_mode='vmm'):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        self.feature_size = num_feat

        # default settings
        layer_channel = [int(c / c_prune_rate) for c in [128, 256]]
        in_channel_list = [in_channel, int(128/c_prune_rate) + 3, int(256/c_prune_rate) + 3]
        ngroup_list = [256, 128, None]
        radius_list = [radius, radius_multiple * radius, None]
        nsample_list = [32, 64, None]

        self.sa = nn.ModuleList()
        if len(iter) > 1:
            for l, layer in enumerate(iter[:-1]):
                mlp = [int(layer_channel[l]) for i in range(layer)]
                sa = NewGraphSetAbstraction(npoint=ngroup_list[l], radius=radius_list[l], nsample=nsample_list[l],
                                    in_channel=in_channel_list[l], mlp=mlp, group_all=False,
                                    noise=noise, quantize=quantize, distancing=distancing,
                                    act=act, hardweight=hardweight, mode=hard_mode)
                self.sa.append(sa)
        mlp_last = [int(self.feature_size / c_prune_rate) for i in range(iter[-1])]
        sa = NewGraphSetAbstraction(npoint=ngroup_list[-1], radius=radius_list[-1], nsample=nsample_list[-1],
                                    in_channel=in_channel_list[l + 1], mlp=mlp_last, group_all=True,
                                    noise=noise, quantize=quantize, distancing=distancing,
                                    act=act, hardweight=hardweight, mode=hard_mode)
        self.sa.append(sa)

        Linear = NoiseLinear
        self.fc = nn.ModuleList()
        self.bn, self.drop = nn.ModuleList(), nn.ModuleList()
        fc_channels = [int(c / c_prune_rate) for c in [self.feature_size, 512, 256]]
        if num_fc > 1:
            for i, c in enumerate(fc_channels[:-1]):
                fc = Linear(fc_channels[i], fc_channels[i + 1], noise=noise)
                self.fc.append(fc)
                self.bn.append(nn.BatchNorm1d(fc_channels[i + 1]))
                self.drop.append(nn.Dropout(0.4))
            fc_out = Linear(fc_channels[-1], num_class)     # the readout should be noise free
        elif num_fc == 1:
            fc_out = Linear(fc_channels[0], num_class)      # the readout should be noise free
        self.fc.append(fc_out)

        self.noise = noise
        self.c_prune_rate = c_prune_rate

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            points = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            points = None

        # with torch.no_grad():
        for i, sa in enumerate(self.sa):
            xyz, points = sa(xyz, points)
            points = F.relu(points)

        x = points.view(B, int(self.feature_size / self.c_prune_rate))

        if len(self.fc) > 1:
            for fc, bn, drop in zip(self.fc[:-1], self.bn, self.drop):
                x = drop(F.relu(bn(fc(x))))
        x = self.fc[-1](x)
        x = F.log_softmax(x, -1)
        return x, points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.model_utils import FeaturePropagation, NewGraphSetAbstraction


class get_model(nn.Module):
    def __init__(self, num_classes, noise=0,
                 normal_channel=False, c_prune_rate=1,
                 feat1=128, num_feat=1024, num_fc=1,
                #  r0=0.15, r1=0.3,
                 r0=0.1, r1=0.3, quant_bit=6,
                 hardweight=None, hard_mode=None):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        layer_c = [int(c / c_prune_rate) for c in [feat1, 256, num_feat]]
        self.sa1 = NewGraphSetAbstraction(npoint=512, radius=r0, nsample=32, in_channel=6+additional_channel,
                                          mlp=[layer_c[0]], group_all=False, noise=noise, mode=hard_mode)
        self.sa2 = NewGraphSetAbstraction(npoint=128, radius=r1, nsample=64, in_channel=layer_c[0]+ 3,
                                          mlp=[layer_c[1]], group_all=False, noise=noise, mode=hard_mode)
        self.sa3 = NewGraphSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=layer_c[1] + 3,
                                          mlp=[layer_c[2]], group_all=True, noise=noise, mode=hard_mode)
        self.fp3 = FeaturePropagation(in_channel=layer_c[2] + layer_c[1], mlp=[layer_c[1]],
                                      noise=noise, mode=hard_mode)
        self.fp2 = FeaturePropagation(in_channel=layer_c[1] + layer_c[0], mlp=[layer_c[0]], 
                                      noise=noise, mode=hard_mode)
        self.fp1 = FeaturePropagation(in_channel=layer_c[0]+16+6+additional_channel, mlp=[layer_c[0]],
                                      noise=noise, mode=hard_mode)
        self.bn1 = nn.BatchNorm1d(layer_c[0])
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(layer_c[0], num_classes, 1)  # [feat1, 50]

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz

        # with torch.no_grad():
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot, l0_xyz, l0_points],1), l1_points)

        # FC layers
        # feat =  F.relu(self.bn1(self.conv1(l0_points)))
        # x = self.drop1(feat)
        x = self.conv2(l0_points)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss

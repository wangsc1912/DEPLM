from turtle import forward
from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
# import backend


class NoiseModule(nn.Module):
    def __init__(self):
        super(NoiseModule, self).__init__()

    def add_noise(self, weight, noise):
        # one = 1.cuda()
        new_w = weight + weight * noise * torch.randn_like(weight)
        # new_w = nn.
        # return nn.Parameter(new_w, requires_grad=False).to(weight.device)
        return new_w.to(weight.device)

    def gen_noise(self, weight, noise):
        new_w = weight * noise * torch.randn_like(weight)
        return new_w.to(weight.device)


class NoiseLinear(NoiseModule):
    def __init__(self, in_features, out_features, sample_noise=False, noise=0, is_train=True, is_hard=False):
        super(NoiseLinear, self).__init__()
        self.noise = noise
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.sample_noise = sample_noise
        self.is_train = is_train
        self.hardware = is_hard

    def forward(self, x):
        if not self.noise:
            return self.linear(x)
        # elif self.is_train:
        #     return self.linear(x) + self.noised_foward(x)
        # else:
        #     return self.linear(x) + self.noised_inference(x)
        else:
            return self.linear(x) + self.noised_forward(x)

    def noised_inference(self, x):
        origin_weight = self.linear.weight
        batch, n_points = x.size()

        x_new = torch.zeros(batch, self.out_features).to(x.device)
        for i in range(batch):
            noise_weight = nn.Parameter(self.gen_noise(origin_weight, self.noise), requires_grad=False)
            x_i = torch.matmul(noise_weight, x[i])
            x_new[i] = x_i
            del noise_weight, x_i

        x_new = x_new.reshape(batch, self.out_features)
        return x_new

    def noised_forward(self, x):
        '''
        forward propagation with noise
        '''
        # x shape: (batch_size, n_points)
        batch_size, n_points = x.size()
        # x = x.reshape(1, -1)

        origin_weight = self.linear.weight
        # x_new = torch.zeros(self.out_features, batch_size*n_points).to(x.device)
        x_new = torch.zeros(batch_size, self.out_features).to(x.device)

        for i in range(x.shape[0]):
            noise_weight = self.gen_noise(origin_weight, self.noise).detach()
            x_i = torch.matmul(noise_weight, x[i, :].unsqueeze(1))
            x_new[i, :] = x_i.squeeze()    # (batch_size, out_features)
            del noise_weight, x_i

        return x_new


class NoiseConv(NoiseModule):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, sample_noise=False, noise=0,
                 hard_weight=None, mode='batch', quant=6):
        super(NoiseConv, self).__init__()
        self.noise = noise
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_noise = sample_noise
        self.hard_weight = hard_weight

        # quant
        self.quant_bit = quant
        self.quant_levels = 2 ** quant - 1
        self.quant_base = 2 ** torch.arange(quant).cuda()
        self.scaling, self.b = 1, 1

        assert mode in ['vmm', 'batch', None], 'mode must be vmm or batch or None!'
        self.mode = mode

        if mode == 'vmm':
            self.code = self.hard_weight.register(layer=self.conv, bias=True)
        else:
            self.code = None

    def forward(self, x):
        if self.hard_weight is None:
            if not self.noise:
                return self.conv(x)
            else:
                return self.conv(x) + self.noised_forward(x)
        else:
            # ori_x = x
            x = self.channel_wise_quantize(x)

                # for b in range(x.shape[-1]):
                    # out_b = self.hardware_inference(x[:, :, :, :, b])
                    # out_b = F.conv2d(x[:, :, :, :, b], self.conv.weight.to(torch.float64), stride=1, padding=0)
                    # out.append(out_b) 
            out = self.hardware_inference(x)
            return out.to(torch.float)

    def quantize(self, x, quant_bits=6):
        # quantize the input x into 6 bits binary numbers
        x = x.detach()
        sample_max = torch.max(x)
        # sample_max = torch.max(sample_max, dim=2, keepdim=True)[0]
        # sample_max = torch.max(sample_max, dim=3, keepdim=True)[0]
        sample_min = torch.min(x)
        # sample_min = torch.min(sample_min, dim=3, keepdim=True)[0]
        # scaling
        self.scaling = (sample_max - sample_min) / self.quant_levels
        self.b = sample_min

        # cast to integers
        x_int = torch.round((x - sample_min) / self.scaling).to(torch.int8)
        x_binary = x_int.unsqueeze(-1).bitwise_and(self.quant_base).ne(0).byte()
        x_binary = x_binary.to(torch.float)
        return x_binary

    def channel_wise_quantize(self, x):
        # quantize the input x into 6 bits binary numbers channel-wisely
        x = x.detach() # (batch_size, channels, nsamples, npoints)
        # scaling
        # a = torch.max(torch.abs(x), dim=1, keepdim=True)[0]
        channel_max = torch.max(x, dim=1, keepdim=True)[0]
        channel_min = torch.min(x, dim=1, keepdim=True)[0]
        self.scaling = (channel_max - channel_min) / self.quant_levels
        self.scaling = self.scaling.to(torch.float64)
        self.b = channel_min.to(torch.float64)

        # cast to integers
        x_int = torch.round((x - channel_min) / self.scaling).to(torch.int64)
        x_binary = x_int.unsqueeze(-1).bitwise_and(self.quant_base).ne(0).byte()
        x_binary = x_binary.to(torch.float64)
        return x_binary

    def dequantize(self, x, quant):
        x = x.detach()
        x = x * (2 ** quant)
        return x

    def dequantize_channel_wise(self, out: list):
        batch, out_c, h, w = out[0].shape
        conv_b = self.conv.bias.reshape(1, -1, 1, 1).repeat(batch, 1, h, w)
        out = [out[i] * self.quant_base[i] for i in range(len(out))]
        out = sum(out)
        # out = out * self.scaling + torch.outer(self.conv.weight.sum(dim=1), self.b)
        out = out * self.scaling 
        out += self.conv.weight.sum(dim=1) * self.b.repeat(1, out_c, 1, 1) 
        out += conv_b 
        out -= (self.quant_levels - 0) * self.scaling * conv_b
        return out

    def hardware_inference(self, x):
        x = x.detach()

        batch_size, in_features, nsamples, npoints, quant_bit = x.size()
        x_bits = []
        #TODO: only support quantization now.
        # to support for no quantization situation.
        for b in range(quant_bit):
            x_bit = x[:, :, :, :, b]
            if self.mode == 'batch':
                x_bit = F.conv2d(x[:, :, :, :, b], self.conv.weight.to(torch.float64), stride=1, padding=0)
                weight, bias = self.conv.weight, self.conv.bias
            if self.mode == 'vmm':
                # x = x.reshape(-1, in_features, 1, 1)
                x_new = []
                for i in range(x_bit.shape[0]):
                    x_sample = []
                    for j in range(x_bit.shape[2]):
                        x_group = []
                        for z in range(x_bit.shape[3]):
                            # xi = x_bit[i, :, j-1:j+1, z-1:z+1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)   # for 3x3 conv
                            xi = x_bit[i, :, j, z].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                            weight = self.hard_weight(self.conv, self.code)
                            if type(weight) is tuple:
                                weight, bias = weight
                            else:
                                bias = self.conv.bias
                            weight, bias = weight.to(x.device).to(x.dtype), bias.to(x.device).to(x.dtype)
                            if (self.conv.weight - weight).mean() > 2:
                                print(f'large error {(self.conv.weight - weight).mean()} at {i, j, z}')
                            # move the bias out
                            xi = F.conv2d(xi, weight, stride=1, padding=0)
                            x_group.append(xi)
                        x_group = torch.cat(x_group, 3) # [1, out_feat, 1, npoints]
                        x_sample.append(x_group)
                    x_sample = torch.cat(x_sample, 2) # [1, out-feat, nsamples, npoints]
                    x_new.append(x_sample)

                x_bit = torch.cat(x_new, 0) # [batch, out_feat, nsamples, npoints]
            x_bits.append(x_bit)

        # dequant
        out_bits = [x_bits[i] * self.quant_base[i] for i in range(len(x_bits))]
        out_sum = sum(out_bits)
        out_w = out_sum * self.scaling 
        out_w += F.conv2d(self.b * torch.ones_like(x[:, :, :, :, 0]), weight.to(torch.float64), bias=bias.to(torch.float64))
        return out_w

    def noised_inference(self, x):
        origin_weight = self.conv.weight.squeeze()
        batch, nsamples, npoints = x.shape[0], x.shape[2], x.shape[3]

        x = x.reshape(batch, self.in_channels, -1, 1).squeeze()    # [channel, number of points]
        x_new = torch.zeros(batch, self.out_channels, x.shape[2], 1).to(x.device)
        for i in range(batch):
            noised_weight = self.gen_noise(origin_weight, self.noise).detach()

            x_i = torch.matmul(noised_weight, x[i])
            x_new[i] = x_i.unsqueeze(-1)
        del noised_weight, x_i
        x_new = x_new.reshape(batch, self.out_channels, nsamples, npoints)
        return x_new.detach()

    def noised_forward(self, x):
        '''
        forward propagation with noise
        '''
        x = x.detach()

        # x shape: (batch_size, in_features, nsamples, npoints)
        # n_points: number of centroids.
        # nsamples: number of points in the neigbor of each centroid.
        batch_size, in_features, nsamples, npoints = x.size()
        # x = x.reshape(1, in_features, 1, -1)
        x = x.reshape(-1, in_features, 1, 1)

        origin_weight = self.conv.weight
        x_new = torch.zeros(x.shape[0], self.out_channels, 1, 1)

        for i in range(x.shape[0]):
            noise_weight= self.gen_noise(origin_weight, self.noise).detach()#.suqeeze()# .detach()
            noise_weight = noise_weight.squeeze()
            # noise_conv = noise_weight
            # del noise_weight
            x_i = x[i, :, :, :].squeeze(-1)#.unsqueeze(-1)
            x_i = torch.matmul(noise_weight, x_i)
            x_new[i, :, :, :] = x_i.unsqueeze(-1)    # (batch_size, out_features)
            del noise_weight, x_i

        x_new = x_new.reshape(batch_size, self.out_channels, nsamples, npoints)
        return x_new.to(x.device).detach()


class NoiseConv1d(NoiseModule):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, sample_noise=False, noise=0,
                 hard_weight=None, mode='batch', quant=6):
        super(NoiseConv1d, self).__init__()
        self.noise = noise
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sample_noise = sample_noise
        self.hard_weight = hard_weight

        # quant
        self.quant_bit = quant
        self.quant_levels = 2 ** quant - 1
        self.quant_base = 2 ** torch.arange(quant).cuda()
        self.scaling, self.b = 0, 0

        assert mode in ['vmm', 'sample', 'batch', None], 'mode must be vmm or batch or sample or None!'
        self.mode = mode

        if mode == 'vmm':
            self.code = self.hard_weight.register(layer=self.conv, bias=True)
        else:
            self.code = None

    def forward(self, x):
        if self.hard_weight is None:
            if not self.noise:
                return self.conv(x)
            else:
                return self.conv(x) + self.noised_forward(x)
        else:
            ori_x = x
            x = self.channel_wise_quantize(x)

            out = self.hardware_inference(x)
            return out.to(torch.float)

    def channel_wise_quantize(self, x):
        # quantize the input x into 6 bits binary numbers channel-wisely
        x = x.detach() # (batch_size, channels, nsamples, npoints)
        # scaling
        channel_max = torch.max(x, dim=1, keepdim=True)[0]
        channel_min = torch.min(x, dim=1, keepdim=True)[0]
        self.scaling = (channel_max - channel_min) / self.quant_levels
        self.scaling = self.scaling.to(torch.float64)
        self.b = channel_min.to(torch.float64)

        # cast to integers
        x_int = torch.round((x - channel_min) / self.scaling).to(torch.int64)
        x_binary = x_int.unsqueeze(-1).bitwise_and(self.quant_base).ne(0).byte()
        x_binary = x_binary.to(torch.float64)
        return x_binary

    def hardware_inference(self, x):
        x = x.detach()

        batch_size, in_features, npoints, quant_bit = x.size()
        x_bits = []
        #TODO: only support quantization now.
        # to support for no quantization situation.
        for b in range(quant_bit):
            x_bit = x[:, :, :, b]
            if self.mode == 'batch':
                x_bit = F.conv1d(x[:, :, :, b], self.conv.weight.to(torch.float64), stride=1, padding=0)
                weight, bias = self.conv.weight, self.conv.bias
            if self.mode == 'vmm':
                # x = x.reshape(-1, in_features, 1, 1)
                x_new = []
                # batch
                for i in range(x_bit.shape[0]):
                    x_sample = []
                    # spatial
                    for j in range(x_bit.shape[2]):
                        xi = x_bit[i, :, j].unsqueeze(0).unsqueeze(-1)
                        xi = backend.conv(xi, self.conv, self.code, stride=1, padding=0)
                        x_sample.append(xi)
                    x_sample = torch.cat(x_sample, 2) # [1, out-feat, nsamples]
                    x_new.append(x_sample)

                x_bit = torch.cat(x_new, 0) # [batch, out_feat, nsamples, npoints]
            x_bits.append(x_bit)

        # dequant
        out_bits = [x_bits[i] * self.quant_base[i] for i in range(len(x_bits))]
        out_sum = sum(out_bits)
        out_w = out_sum * self.scaling 
        out_w += F.conv1d(self.b * torch.ones_like(x[:, :, :, 0]), weight.to(torch.float64), bias=bias.to(torch.float64))
        return out_w

    def noised_forward(self, x):
        '''
        forward propagation with noise
        '''
        x = x.detach()

        # x shape: (batch_size, in_features, nsamples, npoints)
        # n_points: number of centroids.
        # nsamples: number of points in the neigbor of each centroid.
        batch_size, in_features, npoints = x.size()
        origin_weight = self.conv.weight

        if self.mode == 'sample':
            x = x.reshape(-1, in_features, 1, 1)
            x_new = torch.zeros(x.shape[0], self.out_channels, 1, 1)

            for i in range(x.shape[0]):
                noise_weight= self.gen_noise(origin_weight, self.noise).detach()#.suqeeze()# .detach()
                noise_weight = noise_weight.squeeze()
                x_i = x[i, :, :].squeeze(-1)#.unsqueeze(-1)
                x_i = torch.matmul(noise_weight, x_i)
                x_new[i, :, :] = x_i.unsqueeze(-1)    # (batch_size, out_features)
                del noise_weight, x_i
            x_new = x_new.reshape(batch_size, self.out_channels, npoints)

        elif self.mode == 'batch':
            noise_weight= self.gen_noise(origin_weight, self.noise).detach()
            x_new = F.conv1d(x, noise_weight)

        return x_new.to(x.device).detach()

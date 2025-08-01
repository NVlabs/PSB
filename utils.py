# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist


def all_reduce(x):
    x_reduce = x.clone()
    dist.all_reduce(x_reduce)
    return x_reduce


def linear_warmup(step, start_value, final_value, start_step, final_step):

    assert start_value <= final_value
    assert start_step <= final_step

    if step < start_step:
        value = start_value
    elif step >= final_step:
        value = final_value
    else:
        a = final_value - start_value
        b = start_value
        progress = (step + 1 - start_step) / (final_step - start_step)
        value = a * progress + b

    return value


def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    m = nn.Linear(in_features, out_features, bias)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)

    if bias:
        nn.init.zeros_(m.bias)

    return m


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
           dilation=1, groups=1, bias=True, padding_mode='zeros',
           weight_init='xavier', gain=1.):
    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  dilation, groups, bias, padding_mode)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)

    if bias:
        nn.init.zeros_(m.bias)

    return m


class NormalizePixel(nn.Module):
    def __init__(self):
        super().__init__()

        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]), requires_grad=False)

    def forward(self, pixel):
        *ORI, _ = pixel.shape
        pixel = pixel.flatten(end_dim=-2)
        pixel = (pixel - self.mean[None, :]) / self.std[None, :]
        pixel = pixel.reshape(*ORI, -1)
        return pixel

    def inverse(self, pixel):
        *ORI, _ = pixel.shape
        pixel = pixel.flatten(end_dim=-2)
        pixel = pixel * self.std[None, :] + self.mean[None, :]
        pixel = pixel.reshape(*ORI, -1)
        return pixel


class OctavesScalarEncoder(nn.Module):
    def __init__(self, D, max_period):
        super().__init__()
        assert D % 2 == 0
        self.D = D
        self.multipliers = nn.Parameter((2 ** torch.arange(self.D // 2).float()) * 2 * math.pi / max_period, requires_grad=False)

    def forward(self, scalars):
        """

        :param scalars: *ORI
        :param D:
        :return: *ORI, D
        """
        ORI = scalars.shape
        scalars = scalars.flatten()  # B

        x = scalars[:, None] * self.multipliers[None, :]  # B, D // 2
        x = math.sqrt(2) * torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # B, D
        x = x.reshape(*ORI, self.D)  # *ORI, D

        return x

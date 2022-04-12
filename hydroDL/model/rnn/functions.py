import numpy as np
import torch
import torch.nn.functional as F


def UH_conv(x, UH, viewmode=1):
    # UH is a vector indicating the unit hydrograph
    # the convolved dimension will be the last dimension
    # UH convolution is
    # Q(t)=\integral(x(\tao)*UH(t-\tao))d\tao
    # conv1d does \integral(w(\tao)*x(t+\tao))d\tao
    # hence we flip the UH
    # https://programmer.group/pytorch-learning-conv1d-conv2d-and-conv3d.html
    # view
    # x: [batch, var, time]
    # UH:[batch, var, uhLen]
    # batch needs to be accommodated by channels and we make use of groups
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    # https://pytorch.org/docs/stable/nn.functional.html

    mm = x.shape
    nb = mm[0]
    m = UH.shape[-1]
    padd = m - 1
    if viewmode == 1:
        xx = x.view([1, nb, mm[-1]])
        w = UH.view([nb, 1, m])
        groups = nb

    y = F.conv1d(
        xx, torch.flip(w, [2]), groups=groups, padding=padd, stride=1, bias=None
    )
    y = y[:, :, 0:-padd]
    return y.view(mm)


def UH_gamma(a, b, lenF=10):
    # UH. a [time (same all time steps), batch, var]
    m = a.shape
    w = torch.zeros([lenF, m[1], m[2]])
    aa = (
        F.relu(a[0:lenF, :, 0]).view([lenF, m[1], m[2]]) + 0.1
    )  # minimum 0.1. First dimension of a is repeat
    theta = F.relu(b[0:lenF, :, 0]).view([lenF, m[1], m[2]]) + 0.5  # minimum 0.5
    t = torch.arange(0.5, lenF * 1.0).view([lenF, 1, 1]).repeat([1, m[1], m[2]])
    t = t.cuda(aa.device)
    denom = (aa.lgamma().exp()) * (theta ** aa)
    mid = t ** (aa - 1)
    right = torch.exp(-t / theta)
    w = 1 / denom * mid * right
    w = w / w.sum(0)  # scale to 1 for each UH

    return w

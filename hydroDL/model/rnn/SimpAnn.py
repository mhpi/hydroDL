import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class SimpAnn(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize):
        super(SimpAnn, self).__init__()
        self.hiddenSize = hiddenSize
        self.i2h = nn.Linear(nx, hiddenSize)
        self.h2h = nn.Linear(hiddenSize, hiddenSize)
        self.h2o = nn.Linear(hiddenSize, ny)
        self.ny = ny
        self.name = "SimpAnn"
        self.is_legacy = True

    def forward(self, x):
        ht = F.relu(self.i2h(x))
        ht2 = F.relu(self.h2h(ht))
        out = F.relu(self.h2o(ht2))
        return out

import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from hydroDL.model.dropout import DropMask, createMask
from hydroDL.model import rnn
import csv
import numpy as np


class Hymod(torch.nn.Module):
    """Simple 5 parameter model"""

    def __init__(self, *, a, b, cmax, rq, rs, s, slow, fast):
        """Initiate a Hymod instance"""
        super(Hymod, self).__init__()
        self.a = a  # percentage of quickflow
        self.b = b  # shape of Pareto ditribution
        self.cmax = cmax  # maximum storage capacity
        self.rq = rq  # quickflow time constant
        self.rs = rs  # slowflow time constant
        self.smax = self.cmax / (1.0 + self.b)
        self.s = s  # soil moisture
        self.slow = slow  # slowflow reservoir
        self.fast = fast  # fastflow reservoirs
        self.error = 0
        self.name = "Hymod"
        self.is_legacy = True

    def __repr__(self):
        bstr = "a:{!r}".format(self.a)
        bstr += " b:{!r}".format(self.b)
        bstr += " cmax:{!r}".format(self.cmax)
        bstr += " rq:{!r}".format(self.rq)
        bstr += " rs:{!r}".format(self.rs)
        bstr += " smax:{!r}\n".format(self.smax)
        bstr += "s:{!r}".format(self.s)
        bstr += " slow:{!r}".format(self.slow)
        bstr += " fast:{!r}".format(self.fast)
        bstr += " error:{!r}".format(self.error)
        return bstr

    def advance(self, P, PET):
        if self.s > self.smax:
            self.error += self.s - 0.999 * self.smax
            self.s = 0.999 * self.smax

        cprev = self.cmax * (
            1 - np.power((1 - ((self.b + 1) * self.s / self.cmax)), (1 / (self.b + 1)))
        )
        ER1 = np.maximum(P + cprev - self.cmax, 0.0)  # effective rainfal part 1
        P -= ER1
        dummy = np.minimum(((cprev + P) / self.cmax), 1)
        s1 = (self.cmax / (self.b + 1)) * (
            1 - np.power((1 - dummy), (self.b + 1))
        )  # new state
        ER2 = np.maximum(P - (s1 - self.s), 0)  # effective rainfall part 2
        evap = np.minimum(
            s1, s1 / self.smax * PET
        )  # actual ET is linearly related to the soil moisture state
        self.s = s1 - evap  # update state
        UQ = ER1 + self.a * ER2  # quickflow contribution
        US = (1 - self.a) * ER2  # slowflow contribution
        for i in range(3):
            self.fast[i] = (1 - self.rq) * self.fast[i] + (
                1 - self.rq
            ) * UQ  # forecast step
            UQ = (self.rq / (1 - self.rq)) * self.fast[i]
        self.slow = (1 - self.rs) * self.slow + (1 - self.rs) * US
        US = (self.rs / (1 - self.rs)) * self.slow
        Q = UQ + US
        return Q, evap

    def getfast(self):
        return self.fast

    def getslow(self):
        return self.slow

    def getsoilmoisture(self):
        return self.s

    def getparams(self):
        return self.a, self.b, self.cmax, self.rq.self.rs

    def geterror(self):
        return self.error

    def setfast(self, fast):
        self.fast = fast

    def setslow(self, slow):
        self.slow = slow

    def setsoilmoisture(self, s):
        self.s = s

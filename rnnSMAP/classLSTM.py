
import collections
import argparse
import rnnSMAP
import torch
from . import kuaiLSTM


class optLSTM(collections.OrderedDict):
    def __init__(self, **kw):
        # dataset
        self['rootDB'] = rnnSMAP.kPath['DB_L3_Global']
        self['rootOut'] = rnnSMAP.kPath['Out_L3_Global']
        self['gpu'] = 1
        self['out'] = 'test'
        self['train'] = 'Globalv8f1'
        self['var'] = 'varLst_soilM'
        self['varC'] = 'varConstLst_Noah'
        self['target'] = 'SMAP_AM'
        self['syr'] = 2015
        self['eyr'] = 2016
        self['resume'] = 0
        # model
        self['model'] = 'cudnn'
        self['modelOpt'] = 'tied+relu'
        self['hiddenSize'] = 256
        self['dr'] = 0.5
        self['drMethod'] = 'drX+drH+drW+drC'
        self['rho'] = 30
        self['rhoL'] = 30
        self['rhoP'] = 0
        self['nbatch'] = 100
        self['nEpoch'] = 500
        self['saveEpoch'] = 100
        self['addFlag'] = 0
        self['sigma'] = 0
        if kw.keys() is not None:
            for key in kw:
                if key in self:
                    try:
                        self[key] = type(self[key])(kw[key])
                    except:
                        print('skiped '+key+': wrong type')
                else:
                    print('skiped '+key+': not in argument dict')

    def checkOpt(self):
        if self['model'] == 'cudnn':
            self['modelOpt'] = 'tied+relu'
            self['drMethod'] = 'drW'

    def toParser(self):
        parser = argparse.ArgumentParser()
        for key, value in self.items():
            parser.add_argument('--'+key, dest=key,
                                default=value, type=type(value))
        return parser

    def toCmdLine(self):
        cmdStr=''
        for key, value in self.items():
            cmdStr=cmdStr+' --'+key+' '+str(value)
        return cmdStr

    def fromParser(self, parser):
        args = parser.parse_args()
        for key, value in self.items():
            self[key] = getattr(args, key)


def initLSTMstate(ngrid, hiddenSize, gpu, nDim=3):
    if nDim == 3:
        h0 = torch.zeros(
            1, ngrid, hiddenSize, requires_grad=True)
        c0 = torch.zeros(
            1, ngrid, hiddenSize, requires_grad=True)
    if nDim == 2:
        h0 = torch.zeros(
            ngrid, hiddenSize, requires_grad=True)
        c0 = torch.zeros(
            ngrid, hiddenSize, requires_grad=True)
    if gpu > 0:
        h0 = h0.cuda()
        c0 = c0.cuda()
    return h0, c0


class sigmaLoss(torch.nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(sigmaLoss, self).__init__()
        self.reduction = 'elementwise_mean'

    def forward(self, input, target):
        p = input[:, :, 0]
        # s = input[-1, :, 1]
        s = input[:, :, 1]
        t = target[:, :, 0]
        # loss = torch.mean(torch.exp(-s).mul(torch.mul(p-t, p-t))/2, dim=0)+s/2
        loss = torch.mean(torch.exp(-s).mul(torch.mul(p-t, p-t))/2+s/2, dim=0)
        lossMean = torch.mean(loss)
        return lossMean


class torchLSTM(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize):
        super(torchLSTM, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.nLayer = 1
        self.lstm = torch.nn.LSTM(nx, hiddenSize, 1).cuda()
        self.linear = torch.nn.Linear(hiddenSize, ny).cuda()
        self.is_cuda = True
        self.gpu = 1

    def forward(self, x):
        nt = x.size(0)
        ngrid = x.size(1)
        h0, c0 = initLSTMstate(ngrid, self.hiddenSize, self.gpu)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # out = self.dropout(out)
        out = self.linear(out)
        return out


class torchLSTM_cell(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, gpu=1, doReLU=True):
        super(torchLSTM_cell, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.dr = dr
        self.doReLU = doReLU
        self.gpu = gpu

        if doReLU is True:
            self.linearIn = torch.nn.Linear(nx, hiddenSize)
            self.relu = torch.nn.ReLU()
            inputSize = hiddenSize
        else:
            inputSize = nx
        self.lstmcell = torch.nn.LSTMCell(inputSize, hiddenSize)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)

        if gpu > 0:
            self = self.cuda()
            self.is_cuda = True
        else:
            self.is_cuda = False

    def reset_mask(self, x, h):
        self.maskX = kuaiLSTM.createMask(x, self.dr)
        self.maskH = kuaiLSTM.createMask(h, self.dr)
        if self.is_cuda:
            self.maskX = self.maskX.cuda()
            self.maskH = self.maskH.cuda()

    def forward(self, x):
        nt = x.size(0)
        ngrid = x.size(1)
        h0, c0 = initLSTMstate(ngrid, self.hiddenSize, self.gpu, nDim=2)
        ht = h0
        ct = c0

        if self.doReLU is True:
            x0 = self.linearIn(x)
            x0 = self.relu(x0)
        else:
            x0 = x

        output = []
        if self.dr > 0 and self.training is True:
            self.reset_mask(x0[0], h0)

        for i in range(0, nt):
            xt = x0[i]
            if self.dr > 0 and self.training is True:
                xt = kuaiLSTM.dropMask.apply(xt, self.maskX, True)
                ht = kuaiLSTM.dropMask.apply(ht, self.maskH, True)
            ht, ct = self.lstmcell(xt, (ht, ct))
            output.append(ht)
        outView = torch.cat(output, 0).view(nt, *output[0].size())

        out = self.linearOut(outView)
        return out


class localLSTM_cuDNN(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, doReLU=True):
        super(localLSTM_cuDNN, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.nLayer = 1
        self.doReLU = doReLU

        if doReLU is True:
            self.linearIn = torch.nn.Linear(nx, hiddenSize)
            self.relu = torch.nn.ReLU()
            inputSize = hiddenSize
        else:
            inputSize = nx
        self.lstm = kuaiLSTM.cudnnLSTM(
            inputSize=inputSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.is_cuda = True
        self.gpu = 1

    def forward(self, x):
        if self.doReLU is True:
            x0 = self.linearIn(x)
            x0 = self.relu(x0)
        else:
            x0 = x
        outLSTM, (hn, cn) = self.lstm(x0)
        out = self.linearOut(outLSTM)
        return out


class localLSTM_slow(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, gpu=1, drMethod, doReLU=True, doTied=True):
        super(modelLSTM_local, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.nLayer = 1
        self.doReLU = doReLU
        self.doTied = doTied

        if doReLU is True:
            self.linearIn = torch.nn.Linear(nx, hiddenSize)
            self.relu = torch.nn.ReLU()
            inputSize = hiddenSize
        else:
            inputSize = nx

        if doTied is True:
            self.lstm = kuaiLSTM.tiedLSTMcell(
                inputSize=inputSize, hiddenSize=hiddenSize, dr=dr, gpu=gpu, drMethod=drMethod)
        else:
            self.lstm = kuaiLSTM.untiedLSTMcell(
                inputSize=inputSize, hiddenSize=hiddenSize, dr=dr, gpu=gpu, drMethod=drMethod)

        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = gpu
        if gpu > 0:
            self = self.cuda()
            self.is_cuda = True
        else:
            self.is_cuda = False

    def forward(self, x):
        nt = x.size(0)
        ngrid = x.size(1)
        h0, c0 = initLSTMstate(ngrid, self.hiddenSize, self.gpu, nDim=2)
        ht = h0
        ct = c0

        if self.doReLU is True:
            x0 = self.linearIn(x)
            x0 = self.relu(x0)
        else:
            x0 = x

        output = []
        for i in range(0, nt):
            ht, ct = self.lstm(x0[i], (ht, ct))
            output.append(ht)
        outView = torch.cat(output, 0).view(nt, *output[0].size())

        out = self.linearOut(outView)
        return out

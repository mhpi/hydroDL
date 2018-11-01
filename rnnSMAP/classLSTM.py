
import collections
import argparse
import rnnSMAP
import torch
from . import kuaiLSTM
import numpy as np


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
        self['loss'] = 'mse'
        self['lossPrior'] = 'gauss'
        if kw.keys() is not None:
            for key in kw:
                if key in self:
                    try:
                        self[key] = type(self[key])(kw[key])
                    except ValueError:
                        print('skiped '+key+': wrong type')
                else:
                    print('skiped '+key+': not in argument dict')
        self.checkOpt()

    def checkOpt(self):
        if self['model'] == 'cudnn':
            self['modelOpt'] = 'tied+relu'
            self['drMethod'] = 'drW'
        if self['loss'] == 'mse':
            self['lossPrior'] = 'gauss'

    def toParser(self):
        parser = argparse.ArgumentParser()
        for key, value in self.items():
            parser.add_argument('--'+key, dest=key,
                                default=value, type=type(value))
        return parser

    def toCmdLine(self):
        cmdStr = ''
        for key, value in self.items():
            cmdStr = cmdStr+' --'+key+' '+str(value)
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
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean', prior=''):
        super(sigmaLoss, self).__init__()
        self.reduction = 'elementwise_mean'
        if prior == '':
            self.prior = None
        else:
            self.prior = prior.split('+')

    def forward(self, input, target):
        p = input[:, :, 0]
        s = input[:, :, 1]
        # s = input[-1, :, 1]
        t = target[:, :, 0]
        loc0 = t == p
        s[loc0] = 1
        # s.detach()
        if self.prior[0] == 'gauss':
            loss = torch.exp(-s).mul(torch.mul(p-t, p-t))/2+s/2
            lossMeanT = torch.mean(loss, dim=0)
        elif self.prior[0] == 'invGamma':
            c1 = float(self.prior[1])
            c2 = float(self.prior[2])
            nt = p.shape[0]
            loss = torch.exp(-s).mul(torch.mul(p-t, p-t)+c2/nt)/2+(1/2+c1/nt)*s
            lossMeanT = torch.mean(loss, dim=0)

        lossMeanB = torch.mean(lossMeanT)
        return lossMeanB


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


class torchLSTM_cell_open_loop(torch.nn.Module): #open loop
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, gpu=1, doReLU=False):
        super(torchLSTM_cell_open_loop, self).__init__()
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
        self.lstmcell = torch.nn.LSTMCell(20, 20)
        self.linearOut = torch.nn.Linear(20, ny)
        self.finallinearOut = torch.nn.Bilinear(30,100, ny)

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
        h0, c0 = initLSTMstate(ngrid, 20, self.gpu, nDim=2)
        ht = h0
        ct = c0
        x0 = x

        output = []
        if self.dr > 0 and self.training is True:
            self.reset_mask(x0[0], h0)

        raj_output=None
        for i in range(0, nt):
            xt = x0[i]
            new_xt = np.array(xt)
            if raj_output is None:
                raj_in = np.array(torch.zeros(28,1))
                final_in = np.append(new_xt, raj_in, axis=1)
            else:
                raj_in = np.array(raj_output.detach().numpy())
                final_in = np.append(new_xt, raj_in, axis=1)
            final_in=torch.tensor(final_in)
            ht, ct = self.lstmcell(final_in, (ht, ct))
            raj_output = self.linearOut(ht)
            output.append(raj_output)
        out = torch.cat(output, 0).view(nt, *output[0].size())
        return out

class torchLSTM_cell_mc(torch.nn.Module): #open loop
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, gpu=0, doReLU=True):
        super(torchLSTM_cell_mc, self).__init__()
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
        #if self.is_cuda:
            #self.maskX = self.maskX.cuda()
            #self.maskH = self.maskH.cuda()

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

        temp_w=0
        for i in range(0, nt):
            xt = x0[i]
            #if self.dr > 0 and self.training is True:
                #xt = kuaiLSTM.dropMask.apply(xt, self.maskX, True)
                #ht = kuaiLSTM.dropMask.apply(ht, self.maskH, True)
            print(nt)
            print(x0.shape)
            print(xt.shape)
            print(ht.shape)
            #print(self.lstmcell.weight_ih.shape)
            #print(self.lstmcell.weight_hh.shape)
            ht, ct = self.lstmcell(xt, (ht, ct))
            print(self.lstmcell.weight_ih[5].shape)
            print(self.lstmcell.weight_hh.shape)
            break
            #temp_w = ht.parameters()
            #dot_np = np.multiply(self.maskH, temp_w)
            #new_temp = np.dot(xt,dot_np)
            #temp_var = np.square(new_temp)
            #vari = np.add(vari,temp_var)
            output.append(ht)
        outView = torch.cat(output, 0).view(nt, *output[0].size())

        out = self.linearOut(outView)
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

    def forward(self, x, doDropMC=False):
        if self.doReLU is True:
            x0 = self.linearIn(x)
            x0 = self.relu(x0)
        else:
            x0 = x
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        out = self.linearOut(outLSTM)
        return out


class localLSTM_slow(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, gpu=1, drMethod, doReLU=True, doTied=True):
        super(localLSTM_slow, self).__init__()
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

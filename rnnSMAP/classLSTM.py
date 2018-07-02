
import collections
import rnnSMAP
import torch
from . import kuaiLSTM


class optLSTM(collections.OrderedDict):
    def __init__(self, **kw):
        # dataset
        self['rootDB'] = rnnSMAP.kPath['DBSMAP_L3_Global']
        self['rootOut'] = rnnSMAP.kPath['OutSMAP_L3_Global']
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
        self['hiddenSize'] = 256
        self['dr'] = 0.5
        self['drMethod'] = 'drX+drH+drW+drC'
        self['model'] = 'local'
        self['modelOpt'] = 'tied'
        self['rho'] = 30
        self['rhoL'] = 30
        self['rhoP'] = 0
        self['nbatch'] = 100
        self['nEpoch'] = 500
        self['saveEpoch'] = 100
        self['addFlag'] = 0
        if kw.keys() is not None:
            for key in kw:
                if key in self:
                    try:
                        self[key] = type(self[key])(kw[key])
                    except:
                        print('skiped '+key+': wrong type')
                else:
                    print('skiped '+key+': not in argument dict')


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
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, gpu=1):
        super(modelLSTMcell, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.nLayer = 1
        self.lstmcell = torch.nn.LSTMCell(nx, hiddenSize)
        self.linear = torch.nn.Linear(hiddenSize, ny)
        self.dr = dr
        if self.dr > 0:
            self.doDropout = True
        else:
            self.doDropout = False

        if gpu > 0:
            self = self.cuda()
            self.is_cuda = True
        else:
            self.is_cuda = False

    def init_mask(self, x, h):
        self.maskX = x.bernoulli(1-self.dr).div_(1-self.dr)
        self.maskH = h.bernoulli(1-self.dr).div_(1-self.dr)
        self.maskX = self.maskX.detach()
        self.maskH = self.maskH.detach()

        if self.is_cuda:
            self.maskX = self.maskX.cuda()
            self.maskH = self.maskH.cuda()

    def forward(self, x):
        nt = x.size(0)
        ngrid = x.size(1)
        h0, c0 = initLSTMstate(ngrid, self.hiddenSize, self.gpu, nDim=2)

        output = []
        if self.doDropout:
            self.init_mask(x[0], h0)

        for i in range(0, nt):
            x0 = x[i]
            if self.doDropout:
                x0 = x0.mul(self.maskX)
                # h0 = h0.mul(self.maskH)
            ht, ct = self.lstmcell(x0, (h0, c0))
            h0 = ht
            c0 = ct
            output.append(ht)
        outView = torch.cat(output, 0).view(nt, *output[0].size())

        out = self.linear(outView)
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


def initLSTMstate(ngrid, hiddenSize, gpu, nDim=3):
    if nDim == 3:
        if gpu > 0:
            h0 = torch.zeros(
                1, ngrid, hiddenSize, requires_grad=True).cuda()
            c0 = torch.zeros(
                1, ngrid, hiddenSize, requires_grad=True).cuda()
        else:
            h0 = torch.zeros(
                1, ngrid, hiddenSize, requires_grad=True)
            c0 = torch.zeros(
                1, ngrid, hiddenSize, requires_grad=True)
    if nDim == 2:
        if gpu > 0:
            h0 = torch.zeros(
                ngrid, hiddenSize, requires_grad=True).cuda()
            c0 = torch.zeros(
                ngrid, hiddenSize, requires_grad=True).cuda()
        else:
            h0 = torch.zeros(
                ngrid, hiddenSize, requires_grad=True)
            c0 = torch.zeros(
                ngrid, hiddenSize, requires_grad=True)
    return h0, c0

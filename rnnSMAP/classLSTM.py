
import collections
import rnnSMAP
import torch
from . import kuaiLSTM


class optLSTM(collections.OrderedDict):
    def __init__(self, **kw):
        self['rootDB'] = rnnSMAP.kPath['DBSMAP_L3_Global']
        self['rootOut'] = rnnSMAP.kPath['OutSMAP_L3_Global']
        self['gpu'] = 0
        self['out'] = 'test'
        self['train'] = 'Globalv8f1'
        self['var'] = 'varLst_soilM'
        self['varC'] = 'varConstLst_Noah'
        self['target'] = 'SMAP_AM'
        self['syr'] = 2015
        self['eyr'] = 2016
        self['resume'] = 0
        self['hiddenSize'] = 256
        self['rho'] = 30
        self['rhoL'] = 30
        self['rhoP'] = 0
        self['dr'] = 0.5
        self['drMethod'] = 'gal+semeniuta'
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


class modelLSTM(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, gpu=0):
        super(modelLSTM, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.nLayer = 1
        self.lstm = torch.nn.LSTM(nx, hiddenSize, 1)
        self.dropout = torch.nn.Dropout(dr)
        self.linear = torch.nn.Linear(hiddenSize, ny)
        self.gpu = gpu
        if gpu >= 0:
            self = self.cuda(gpu)
            self.is_cuda = True
        else:
            self.is_cuda = False

    def forward(self, x):
        h0, c0 = initLSTMstate(ngrid, self.hiddenSize, self.gpu)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.linear(out)
        return out


class modelLSTMcell(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, gpu=0):
        super(modelLSTMcell, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.nLayer = 1
        self.lstmcell = torch.nn.LSTMCell(nx, hiddenSize)
        self.linear = torch.nn.Linear(hiddenSize, ny)
        self.gpu = gpu
        self.dr = dr
        if self.dr > 0:
            self.doDropout = True
        else:
            self.doDropout = False

        if gpu >= 0:
            self = self.cuda(gpu)
            self.is_cuda = True
        else:
            self.is_cuda = False

    def init_mask(self, x, h):
        self.maskX = x.bernoulli(1-self.dr).div_(1-self.dr)
        self.maskH = h.bernoulli(1-self.dr).div_(1-self.dr)
        self.maskX = self.maskX.detach()
        self.maskH = self.maskH.detach()

        if self.is_cuda:
            self.maskX = self.maskX.cuda(self.gpu)
            self.maskH = self.maskH.cuda(self.gpu)

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


class modelLSTM_Kuai(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, gpu=0, drMethod):
        super(modelLSTM_Kuai, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.nLayer = 1
        self.lstm = kuaiLSTM.untiedLSTMcell(
            xSize=nx, hiddenSize=hiddenSize, dr=dr, gpu=gpu, drMethod=drMethod)
        self.linear = torch.nn.Linear(hiddenSize, ny)
        self.gpu = gpu
        if gpu >= 0:
            self = self.cuda(gpu)
            self.is_cuda = True
        else:
            self.is_cuda = False

    def forward(self, x):
        nt = x.size(0)
        ngrid = x.size(1)
        h0, c0 = initLSTMstate(ngrid, self.hiddenSize, self.gpu, nDim=2)

        ht = h0
        ct = c0
        output = []
        for i in range(0, nt):
            ht, ct = self.lstm(x[i], (ht, ct))
            output.append(ht)
        outView = torch.cat(output, 0).view(nt, *output[0].size())

        out = self.linear(outView)
        return out


def initLSTMstate(ngrid, hiddenSize, gpu, nDim=3):
    if nDim == 3:
        if gpu != -1:
            h0 = torch.zeros(
                1, ngrid, hiddenSize, requires_grad=True).cuda(gpu)
            c0 = torch.zeros(
                1, ngrid, hiddenSize, requires_grad=True).cuda(gpu)
        else:
            h0 = torch.zeros(
                1, ngrid, hiddenSize, requires_grad=True)
            c0 = torch.zeros(
                1, ngrid, hiddenSize, requires_grad=True)
    if nDim == 2:
        if gpu != -1:
            h0 = torch.zeros(
                ngrid, hiddenSize, requires_grad=True).cuda(gpu)
            c0 = torch.zeros(
                ngrid, hiddenSize, requires_grad=True).cuda(gpu)
        else:
            h0 = torch.zeros(
                ngrid, hiddenSize, requires_grad=True)
            c0 = torch.zeros(
                ngrid, hiddenSize, requires_grad=True)
    return h0, c0

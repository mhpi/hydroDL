import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .dropout import DropMask, createMask
from . import cnn
import csv
import numpy as np


class LSTMcell_untied(torch.nn.Module):
    def __init__(self,
                 *,
                 inputSize,
                 hiddenSize,
                 train=True,
                 dr=0.5,
                 drMethod='gal+sem',
                 gpu=0):
        super(LSTMcell_untied, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = inputSize
        self.dr = dr

        self.w_xi = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xf = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xo = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xc = Parameter(torch.Tensor(hiddenSize, inputSize))

        self.w_hi = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_hf = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_ho = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_hc = Parameter(torch.Tensor(hiddenSize, hiddenSize))

        self.b_i = Parameter(torch.Tensor(hiddenSize))
        self.b_f = Parameter(torch.Tensor(hiddenSize))
        self.b_o = Parameter(torch.Tensor(hiddenSize))
        self.b_c = Parameter(torch.Tensor(hiddenSize))

        self.drMethod = drMethod.split('+')
        self.gpu = gpu
        self.train = train
        if gpu >= 0:
            self = self.cuda(gpu)
            self.is_cuda = True
        else:
            self.is_cuda = False
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hiddenSize)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def init_mask(self, x, h, c):
        self.maskX_i = createMask(x, self.dr)
        self.maskX_f = createMask(x, self.dr)
        self.maskX_c = createMask(x, self.dr)
        self.maskX_o = createMask(x, self.dr)

        self.maskH_i = createMask(h, self.dr)
        self.maskH_f = createMask(h, self.dr)
        self.maskH_c = createMask(h, self.dr)
        self.maskH_o = createMask(h, self.dr)

        self.maskC = createMask(c, self.dr)

        self.maskW_xi = createMask(self.w_xi, self.dr)
        self.maskW_xf = createMask(self.w_xf, self.dr)
        self.maskW_xc = createMask(self.w_xc, self.dr)
        self.maskW_xo = createMask(self.w_xo, self.dr)
        self.maskW_hi = createMask(self.w_hi, self.dr)
        self.maskW_hf = createMask(self.w_hf, self.dr)
        self.maskW_hc = createMask(self.w_hc, self.dr)
        self.maskW_ho = createMask(self.w_ho, self.dr)

    def forward(self, x, hidden):
        h0, c0 = hidden
        doDrop = self.training and self.dr > 0.0

        if doDrop:
            self.init_mask(x, h0, c0)

        if doDrop and 'drH' in self.drMethod:
            h0_i = h0.mul(self.maskH_i)
            h0_f = h0.mul(self.maskH_f)
            h0_c = h0.mul(self.maskH_c)
            h0_o = h0.mul(self.maskH_o)
        else:
            h0_i = h0
            h0_f = h0
            h0_c = h0
            h0_o = h0

        if doDrop and 'drX' in self.drMethod:
            x_i = x.mul(self.maskX_i)
            x_f = x.mul(self.maskX_f)
            x_c = x.mul(self.maskX_c)
            x_o = x.mul(self.maskX_o)
        else:
            x_i = x
            x_f = x
            x_c = x
            x_o = x

        if doDrop and 'drW' in self.drMethod:
            w_xi = self.w_xi.mul(self.maskW_xi)
            w_xf = self.w_xf.mul(self.maskW_xf)
            w_xc = self.w_xc.mul(self.maskW_xc)
            w_xo = self.w_xo.mul(self.maskW_xo)
            w_hi = self.w_hi.mul(self.maskW_hi)
            w_hf = self.w_hf.mul(self.maskW_hf)
            w_hc = self.w_hc.mul(self.maskW_hc)
            w_ho = self.w_ho.mul(self.maskW_ho)
        else:
            w_xi = self.w_xi
            w_xf = self.w_xf
            w_xc = self.w_xc
            w_xo = self.w_xo
            w_hi = self.w_hi
            w_hf = self.w_hf
            w_hc = self.w_hc
            w_ho = self.w_ho

        gate_i = F.linear(x_i, w_xi) + F.linear(h0_i, w_hi) + self.b_i
        gate_f = F.linear(x_f, w_xf) + F.linear(h0_f, w_hf) + self.b_f
        gate_c = F.linear(x_c, w_xc) + F.linear(h0_c, w_hc) + self.b_c
        gate_o = F.linear(x_o, w_xo) + F.linear(h0_o, w_ho) + self.b_o

        gate_i = F.sigmoid(gate_i)
        gate_f = F.sigmoid(gate_f)
        gate_c = F.tanh(gate_c)
        gate_o = F.sigmoid(gate_o)

        if doDrop and 'drC' in self.drMethod:
            gate_c = gate_c.mul(self.maskC)

        c1 = (gate_f * c0) + (gate_i * gate_c)
        h1 = gate_o * F.tanh(c1)

        return h1, c1


class LSTMcell_tied(torch.nn.Module):
    def __init__(self,
                 *,
                 inputSize,
                 hiddenSize,
                 mode='train',
                 dr=0.5,
                 drMethod='drX+drW+drC',
                 gpu=1):
        super(LSTMcell_tied, self).__init__()

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr

        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))

        self.drMethod = drMethod.split('+')
        self.gpu = gpu
        self.mode = mode
        if mode == 'train':
            self.train(mode=True)
        elif mode == 'test':
            self.train(mode=False)
        elif mode == 'drMC':
            self.train(mode=False)

        if gpu >= 0:
            self = self.cuda()
            self.is_cuda = True
        else:
            self.is_cuda = False
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def reset_mask(self, x, h, c):
        self.maskX = createMask(x, self.dr)
        self.maskH = createMask(h, self.dr)
        self.maskC = createMask(c, self.dr)
        self.maskW_ih = createMask(self.w_ih, self.dr)
        self.maskW_hh = createMask(self.w_hh, self.dr)

    def forward(self, x, hidden, *, resetMask=True, doDropMC=False):
        if self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = x.size(0)
        h0, c0 = hidden
        if h0 is None:
            h0 = x.new_zeros(batchSize, self.hiddenSize, requires_grad=False)
        if c0 is None:
            c0 = x.new_zeros(batchSize, self.hiddenSize, requires_grad=False)

        if self.dr > 0 and self.training is True and resetMask is True:
            self.reset_mask(x, h0, c0)

        if doDrop is True and 'drH' in self.drMethod:
            h0 = DropMask.apply(h0, self.maskH, True)

        if doDrop is True and 'drX' in self.drMethod:
            x = DropMask.apply(x, self.maskX, True)

        if doDrop is True and 'drW' in self.drMethod:
            w_ih = DropMask.apply(self.w_ih, self.maskW_ih, True)
            w_hh = DropMask.apply(self.w_hh, self.maskW_hh, True)
        else:
            # self.w are parameters, while w are not
            w_ih = self.w_ih
            w_hh = self.w_hh

        gates = F.linear(x, w_ih, self.b_ih) + \
            F.linear(h0, w_hh, self.b_hh)
        gate_i, gate_f, gate_c, gate_o = gates.chunk(4, 1)

        gate_i = torch.sigmoid(gate_i)
        gate_f = torch.sigmoid(gate_f)
        gate_c = torch.tanh(gate_c)
        gate_o = torch.sigmoid(gate_o)

        if self.training is True and 'drC' in self.drMethod:
            gate_c = gate_c.mul(self.maskC)

        c1 = (gate_f * c0) + (gate_i * gate_c)
        h1 = gate_o * torch.tanh(c1)

        return h1, c1


class CudnnLstm(torch.nn.Module):
    def __init__(self, *, inputSize, hiddenSize, dr=0.5, drMethod='drW',
                 gpu=0):
        super(CudnnLstm, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr
        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]
        self.cuda()

        self.reset_mask()
        self.reset_parameters()

    def _apply(self, fn):
        ret = super(CudnnLstm, self)._apply(fn)
        return ret

    def __setstate__(self, d):
        super(CudnnLstm, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]

    def reset_mask(self):
        self.maskW_ih = createMask(self.w_ih, self.dr)
        self.maskW_hh = createMask(self.w_hh, self.dr)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, input, hx=None, cx=None, doDropMC=False, dropoutFalse=False):
        # dropoutFalse: it will ensure doDrop is false, unless doDropMC is true
        if dropoutFalse and (not doDropMC):
            doDrop = False
        elif self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = input.size(1)

        if hx is None:
            hx = input.new_zeros(
                1, batchSize, self.hiddenSize, requires_grad=False)
        if cx is None:
            cx = input.new_zeros(
                1, batchSize, self.hiddenSize, requires_grad=False)

        # cuDNN backend - disabled flat weight
        # handle = torch.backends.cudnn.get_handle()
        if doDrop is True:
            self.reset_mask()
            weight = [
                DropMask.apply(self.w_ih, self.maskW_ih, True),
                DropMask.apply(self.w_hh, self.maskW_hh, True), self.b_ih,
                self.b_hh
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

        # output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
        #     input, weight, 4, None, hx, cx, torch.backends.cudnn.CUDNN_LSTM,
        #     self.hiddenSize, 1, False, 0, self.training, False, (), None)
        if torch.__version__ < "1.8":
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input, weight, 4, None, hx, cx, 2,  # 2 means LSTM
                self.hiddenSize, 1, False, 0, self.training, False, (), None)
        else:
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input, weight, 4, None, hx, cx, 2,  # 2 means LSTM
                self.hiddenSize, 0, 1, False, 0, self.training, False, (), None)
        return output, (hy, cy)

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights]
                for weights in self._all_weights]

class CNN1dkernel(torch.nn.Module):
    def __init__(self,
                 *,
                 ninchannel=1,
                 nkernel=3,
                 kernelSize=3,
                 stride=1,
                 padding=0):
        super(CNN1dkernel, self).__init__()
        self.cnn1d = torch.nn.Conv1d(
            in_channels=ninchannel,
            out_channels=nkernel,
            kernel_size=kernelSize,
            padding=padding,
            stride=stride,
        )

    def forward(self, x):
        output = F.relu(self.cnn1d(x))
        return output

class CudnnLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CudnnLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        # self.drtest = torch.nn.Dropout(p=0.4)

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        x0 = F.relu(self.linearIn(x))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC, dropoutFalse=dropoutFalse)
        # outLSTMdr = self.drtest(outLSTM)
        out = self.linearOut(outLSTM)
        return out


class CNN1dLSTMmodel(torch.nn.Module):
    def __init__(self, *, nx, ny, nobs, hiddenSize,
                 nkernel=(10,5), kernelSize=(3,3), stride=(2,1), dr=0.5, poolOpt=None):
        # two convolutional layer
        super(CNN1dLSTMmodel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.obs = nobs
        self.hiddenSize = hiddenSize
        nlayer = len(nkernel)
        self.features = nn.Sequential()
        ninchan = 1
        Lout = nobs
        for ii in range(nlayer):
            ConvLayer = CNN1dkernel(
                ninchannel=ninchan, nkernel=nkernel[ii], kernelSize=kernelSize[ii], stride=stride[ii])
            self.features.add_module('CnnLayer%d' % (ii + 1), ConvLayer)
            ninchan = nkernel[ii]
            Lout = cnn.calConvSize(lin=Lout, kernel=kernelSize[ii], stride=stride[ii])
            self.features.add_module('Relu%d' % (ii + 1), nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d' % (ii + 1), nn.MaxPool1d(poolOpt[ii]))
                Lout = cnn.calPoolSize(lin=Lout, kernel=poolOpt[ii])
        self.Ncnnout = int(Lout*nkernel[-1]) # total CNN feature number after convolution
        Nf = self.Ncnnout + nx
        self.linearIn = torch.nn.Linear(Nf, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, z, doDropMC=False):
        nt, ngrid, nobs = z.shape
        z = z.view(nt*ngrid, 1, nobs)
        z0 = self.features(z)
        # z0 = (ntime*ngrid) * nkernel * sizeafterconv
        z0 = z0.view(nt, ngrid, self.Ncnnout)
        x0 = torch.cat((x, z0), dim=2)
        x0 = F.relu(self.linearIn(x0))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        out = self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out

class CNN1dLSTMInmodel(torch.nn.Module):
    # Directly add the CNN extracted features into LSTM inputSize
    def __init__(self, *, nx, ny, nobs, hiddenSize,
                 nkernel=(10,5), kernelSize=(3,3), stride=(2,1), dr=0.5, poolOpt=None, cnndr=0.0):
        # two convolutional layer
        super(CNN1dLSTMInmodel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.obs = nobs
        self.hiddenSize = hiddenSize
        nlayer = len(nkernel)
        self.features = nn.Sequential()
        ninchan = 1
        Lout = nobs
        for ii in range(nlayer):
            ConvLayer = CNN1dkernel(
                ninchannel=ninchan, nkernel=nkernel[ii], kernelSize=kernelSize[ii], stride=stride[ii])
            self.features.add_module('CnnLayer%d' % (ii + 1), ConvLayer)
            if cnndr != 0.0:
                self.features.add_module('dropout%d' % (ii + 1), nn.Dropout(p=cnndr))
            ninchan = nkernel[ii]
            Lout = cnn.calConvSize(lin=Lout, kernel=kernelSize[ii], stride=stride[ii])
            self.features.add_module('Relu%d' % (ii + 1), nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d' % (ii + 1), nn.MaxPool1d(poolOpt[ii]))
                Lout = cnn.calPoolSize(lin=Lout, kernel=poolOpt[ii])
        self.Ncnnout = int(Lout*nkernel[-1]) # total CNN feature number after convolution
        Nf = self.Ncnnout + hiddenSize
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=Nf, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, z, doDropMC=False):
        nt, ngrid, nobs = z.shape
        z = z.view(nt*ngrid, 1, nobs)
        z0 = self.features(z)
        # z0 = (ntime*ngrid) * nkernel * sizeafterconv
        z0 = z0.view(nt, ngrid, self.Ncnnout)
        x = F.relu(self.linearIn(x))
        x0 = torch.cat((x, z0), dim=2)
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        out = self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out

class CNN1dLCmodel(torch.nn.Module):
    # add the CNN extracted features into original LSTM input, then pass through linear layer
    def __init__(self, *, nx, ny, nobs, hiddenSize,
                 nkernel=(10,5), kernelSize=(3,3), stride=(2,1), dr=0.5, poolOpt=None, cnndr=0.0):
        # two convolutional layer
        super(CNN1dLCmodel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.obs = nobs
        self.hiddenSize = hiddenSize
        nlayer = len(nkernel)
        self.features = nn.Sequential()
        ninchan = 1 # need to modify the hardcode: 4 for smap and 1 for FDC
        Lout = nobs
        for ii in range(nlayer):
            ConvLayer = CNN1dkernel(
                ninchannel=ninchan, nkernel=nkernel[ii], kernelSize=kernelSize[ii], stride=stride[ii])
            self.features.add_module('CnnLayer%d' % (ii + 1), ConvLayer)
            if cnndr != 0.0:
                self.features.add_module('dropout%d' % (ii + 1), nn.Dropout(p=cnndr))
            ninchan = nkernel[ii]
            Lout = cnn.calConvSize(lin=Lout, kernel=kernelSize[ii], stride=stride[ii])
            self.features.add_module('Relu%d' % (ii + 1), nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d' % (ii + 1), nn.MaxPool1d(poolOpt[ii]))
                Lout = cnn.calPoolSize(lin=Lout, kernel=poolOpt[ii])
        self.Ncnnout = int(Lout*nkernel[-1]) # total CNN feature number after convolution
        Nf = self.Ncnnout + nx
        self.linearIn = torch.nn.Linear(Nf, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, z, doDropMC=False):
        # z = ngrid*nVar add a channel dimension
        ngrid = z.shape[0]
        rho, BS, Nvar = x.shape
        if len(z.shape) == 2: # for FDC, else 3 dimension for smap
            z = torch.unsqueeze(z, dim=1)
        z0 = self.features(z)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0 = z0.view(ngrid, self.Ncnnout).repeat(rho,1,1)
        x = torch.cat((x, z0), dim=2)
        x0 = F.relu(self.linearIn(x))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        out = self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out

class CNN1dLCInmodel(torch.nn.Module):
    # Directly add the CNN extracted features into LSTM inputSize
    def __init__(self, *, nx, ny, nobs, hiddenSize,
                 nkernel=(10,5), kernelSize=(3,3), stride=(2,1), dr=0.5, poolOpt=None, cnndr=0.0):
        # two convolutional layer
        super(CNN1dLCInmodel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.obs = nobs
        self.hiddenSize = hiddenSize
        nlayer = len(nkernel)
        self.features = nn.Sequential()
        ninchan = 1
        Lout = nobs
        for ii in range(nlayer):
            ConvLayer = CNN1dkernel(
                ninchannel=ninchan, nkernel=nkernel[ii], kernelSize=kernelSize[ii], stride=stride[ii])
            self.features.add_module('CnnLayer%d' % (ii + 1), ConvLayer)
            if cnndr != 0.0:
                self.features.add_module('dropout%d' % (ii + 1), nn.Dropout(p=cnndr))
            ninchan = nkernel[ii]
            Lout = cnn.calConvSize(lin=Lout, kernel=kernelSize[ii], stride=stride[ii])
            self.features.add_module('Relu%d' % (ii + 1), nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d' % (ii + 1), nn.MaxPool1d(poolOpt[ii]))
                Lout = cnn.calPoolSize(lin=Lout, kernel=poolOpt[ii])
        self.Ncnnout = int(Lout*nkernel[-1]) # total CNN feature number after convolution
        Nf = self.Ncnnout + hiddenSize
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = CudnnLstm(
            inputSize=Nf, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1

    def forward(self, x, z, doDropMC=False):
        # z = ngrid*nVar add a channel dimension
        ngrid, nobs = z.shape
        rho, BS, Nvar = x.shape
        z = torch.unsqueeze(z, dim=1)
        z0 = self.features(z)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0 = z0.view(ngrid, self.Ncnnout).repeat(rho,1,1)
        x = F.relu(self.linearIn(x))
        x0 = torch.cat((x, z0), dim=2)
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        out = self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out

class CudnnInvLstmModel(torch.nn.Module):
    # using cudnnLstm to extract features from SMAP observations
    def __init__(self, *, nx, ny, hiddenSize, ninv, nfea, hiddeninv, dr=0.5, drinv=0.5):
        # two LSTM
        super(CudnnInvLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ninv = ninv
        self.nfea = nfea
        self.hiddeninv = hiddeninv
        self.lstminv = CudnnLstmModel(
            nx=ninv, ny=nfea, hiddenSize=hiddeninv, dr=drinv)
        self.lstm = CudnnLstmModel(
            nx=nfea+nx, ny=ny, hiddenSize=hiddenSize, dr=dr)
        self.gpu = 1

    def forward(self, x, z, doDropMC=False):
        Gen = self.lstminv(z)
        dim = x.shape;
        nt = dim[0]
        invpara = Gen[-1, :, :].repeat(nt, 1, 1)
        x1 = torch.cat((x, invpara), dim=2)
        out = self.lstm(x1)
        # out = rho/time * batchsize * Ntargetvar
        return out


class LstmCloseModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, fillObs=True):
        super(LstmCloseModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx + 1, hiddenSize)
        # self.lstm = CudnnLstm(
        #     inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.lstm = LSTMcell_tied(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr, drMethod='drW')
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        self.fillObs = fillObs

    def forward(self, x, y=None):
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        ht = None
        ct = None
        resetMask = True
        for t in range(nt):
            if self.fillObs is True:
                ytObs = y[t, :, :]
                mask = ytObs == ytObs
                yt[mask] = ytObs[mask]
            xt = torch.cat((x[t, :, :], yt), 1)
            x0 = F.relu(self.linearIn(xt))
            ht, ct = self.lstm(x0, hidden=(ht, ct), resetMask=resetMask)
            yt = self.linearOut(ht)
            resetMask = False
            out[t, :, :] = yt
        return out


class AnnModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize):
        super(AnnModel, self).__init__()
        self.hiddenSize = hiddenSize
        self.i2h = nn.Linear(nx, hiddenSize)
        self.h2h = nn.Linear(hiddenSize, hiddenSize)
        self.h2o = nn.Linear(hiddenSize, ny)
        self.ny = ny

    def forward(self, x, y=None):
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        for t in range(nt):
            xt = x[t, :, :]
            ht = F.relu(self.i2h(xt))
            ht2 = self.h2h(ht)
            yt = self.h2o(ht2)
            out[t, :, :] = yt
        return out


class AnnCloseModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, fillObs=True):
        super(AnnCloseModel, self).__init__()
        self.hiddenSize = hiddenSize
        self.i2h = nn.Linear(nx + 1, hiddenSize)
        self.h2h = nn.Linear(hiddenSize, hiddenSize)
        self.h2o = nn.Linear(hiddenSize, ny)
        self.fillObs = fillObs
        self.ny = ny

    def forward(self, x, y=None):
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        for t in range(nt):
            if self.fillObs is True:
                ytObs = y[t, :, :]
                mask = ytObs == ytObs
                yt[mask] = ytObs[mask]
            xt = torch.cat((x[t, :, :], yt), 1)
            ht = F.relu(self.i2h(xt))
            ht2 = self.h2h(ht)
            yt = self.h2o(ht2)
            out[t, :, :] = yt
        return out


class LstmCnnCond(nn.Module):
    def __init__(self,
                 *,
                 nx,
                 ny,
                 ct,
                 opt=1,
                 hiddenSize=64,
                 cnnSize=32,
                 cp1=(64, 3, 2),
                 cp2=(128, 5, 2),
                 dr=0.5):
        super(LstmCnnCond, self).__init__()

        # opt == 1: cnn output as initial state of LSTM (h0)
        # opt == 2: cnn output as additional output of LSTM
        # opt == 3: cnn output as constant input of LSTM

        if opt == 1:
            cnnSize = hiddenSize

        self.nx = nx
        self.ny = ny
        self.ct = ct
        self.ctRm = False
        self.hiddenSize = hiddenSize
        self.opt = opt

        self.cnn = cnn.Cnn1d(nx=nx, nt=ct, cnnSize=cnnSize, cp1=cp1, cp2=cp2)

        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        if opt == 3:
            self.linearIn = torch.nn.Linear(nx + cnnSize, hiddenSize)
        else:
            self.linearIn = torch.nn.Linear(nx, hiddenSize)
        if opt == 2:
            self.linearOut = torch.nn.Linear(hiddenSize + cnnSize, ny)
        else:
            self.linearOut = torch.nn.Linear(hiddenSize, ny)

    def forward(self, x, xc):
        # x- [nt,ngrid,nx]
        x1 = xc
        x1 = self.cnn(x1)
        x2 = x
        if self.opt == 1:
            x2 = F.relu(self.linearIn(x2))
            x2, (hn, cn) = self.lstm(x2, hx=x1[None, :, :])
            x2 = self.linearOut(x2)
        elif self.opt == 2:
            x1 = x1[None, :, :].repeat(x2.shape[0], 1, 1)
            x2 = F.relu(self.linearIn(x2))
            x2, (hn, cn) = self.lstm(x2)
            x2 = self.linearOut(torch.cat([x2, x1], 2))
        elif self.opt == 3:
            x1 = x1[None, :, :].repeat(x2.shape[0], 1, 1)
            x2 = torch.cat([x2, x1], 2)
            x2 = F.relu(self.linearIn(x2))
            x2, (hn, cn) = self.lstm(x2)
            x2 = self.linearOut(x2)

        return x2


class LstmCnnForcast(nn.Module):
    def __init__(self,
                 *,
                 nx,
                 ny,
                 ct,
                 opt=1,
                 hiddenSize=64,
                 cnnSize=32,
                 cp1=(64, 3, 2),
                 cp2=(128, 5, 2),
                 dr=0.5):
        super(LstmCnnForcast, self).__init__()

        if opt == 1:
            cnnSize = hiddenSize

        self.nx = nx
        self.ny = ny
        self.ct = ct
        self.ctRm = True
        self.hiddenSize = hiddenSize
        self.opt = opt
        self.cnnSize = cnnSize

        if opt == 1:
            self.cnn = cnn.Cnn1d(
                nx=nx + 1, nt=ct, cnnSize=cnnSize, cp1=cp1, cp2=cp2)
        if opt == 2:
            self.cnn = cnn.Cnn1d(
                nx=1, nt=ct, cnnSize=cnnSize, cp1=cp1, cp2=cp2)

        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearIn = torch.nn.Linear(nx + cnnSize, hiddenSize)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)

    def forward(self, x, y):
        # x- [nt,ngrid,nx]
        nt, ngrid, nx = x.shape
        ct = self.ct
        pt = nt - ct

        if self.opt == 1:
            x1 = torch.cat((y, x), dim=2)
        elif self.opt == 2:
            x1 = y

        x1out = torch.zeros([pt, ngrid, self.cnnSize]).cuda()
        for k in range(pt):
            x1out[k, :, :] = self.cnn(x1[k:k + ct, :, :])

        x2 = x[ct:nt, :, :]
        x2 = torch.cat([x2, x1out], 2)
        x2 = F.relu(self.linearIn(x2))
        x2, (hn, cn) = self.lstm(x2)
        x2 = self.linearOut(x2)

        return x2

class CudnnLstmModel_R2P(torch.nn.Module):

    def __init__(self, **arg):
        pass


class CpuLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CpuLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = LSTMcell_tied(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr, drMethod='drW', gpu=-1)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = -1

    def forward(self, x, doDropMC=False):
        # x0 = F.relu(self.linearIn(x))
        # outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        # out = self.linearOut(outLSTM)
        # return out
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1)
        out = torch.zeros(nt, ngrid, self.ny)
        ht = None
        ct = None
        resetMask = True
        for t in range(nt):
            xt = x[t, :, :]
            x0 = F.relu(self.linearIn(xt))
            ht, ct = self.lstm(x0, hidden=(ht, ct), resetMask=resetMask)
            yt = self.linearOut(ht)
            resetMask = False
            out[t, :, :] = yt
        return out


class CudnnInv_HBVModel(torch.nn.Module):

    def __init__(self, **arg):
        pass


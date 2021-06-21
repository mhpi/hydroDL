import torch
import torch.nn as nn
import torch.nn.functional as F


class Cnn1d(nn.Module):
    def __init__(self, *, nx, nt, cnnSize=32, cp1=(64, 3, 2), cp2=(128, 5, 2)):
        super(Cnn1d, self).__init__()
        self.nx = nx
        self.nt = nt
        cOut, f, p = cp1
        self.conv1 = nn.Conv1d(nx, cOut, f)
        self.pool1 = nn.MaxPool1d(p)
        lTmp = int(calConvSize(nt, f, 0, 1, 1) / p)

        cIn = cOut
        cOut, f, p = cp2
        self.conv2 = nn.Conv1d(cIn, cOut, f)
        self.pool2 = nn.MaxPool1d(p)
        lTmp = int(calConvSize(lTmp, f, 0, 1, 1) / p)

        self.flatLength = int(cOut * lTmp)
        self.fc1 = nn.Linear(self.flatLength, cnnSize)
        self.fc2 = nn.Linear(cnnSize, cnnSize)

    def forward(self, x):
        # x- [nt,ngrid,nx]
        x1 = x
        x1 = x1.permute(1, 2, 0)
        x1 = self.pool1(F.relu(self.conv1(x1)))
        x1 = self.pool2(F.relu(self.conv2(x1)))
        x1 = x1.view(-1, self.flatLength)
        x1 = F.relu(self.fc1(x1))
        x1 = self.fc2(x1)

        return x1

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
        # output = self.cnn1d(x)
        return output

class LstmCnn1d(torch.nn.Module):
    # Dense layer > reduce dim > dense
    def __init__(self, *, nx, ny, rho, nkernel=(10,5), kernelSize=(3,3), stride=(2,1), padding=(1,1),
                 dr=0.5, poolOpt=None):
        # two convolutional layer
        super(LstmCnn1d, self).__init__()
        self.nx = nx
        self.ny = ny
        self.rho = rho
        nlayer = len(nkernel)
        self.features = nn.Sequential()
        ninchan = nx
        Lout = rho
        for ii in range(nlayer):
            # First layer: no dimension reduction
            ConvLayer = CNN1dkernel(
                ninchannel=ninchan, nkernel=nkernel[ii], kernelSize=kernelSize[ii],
                stride=stride[ii], padding=padding[ii])
            self.features.add_module('CnnLayer%d' % (ii + 1), ConvLayer)
            ninchan = nkernel[ii]
            Lout = calConvSize(lin=Lout, kernel=kernelSize[ii], stride=stride[ii])
            if poolOpt is not None:
                self.features.add_module('Pooling%d' % (ii + 1), nn.MaxPool1d(poolOpt[ii]))
                Lout = calPoolSize(lin=Lout, kernel=poolOpt[ii])
        self.Ncnnout = int(Lout*nkernel[-1]) # total CNN feature number after convolution


    def forward(self, x, doDropMC=False):
        out = self.features(x)
        # # z0 = (ntime*ngrid) * nkernel * sizeafterconv
        # z0 = z0.view(nt, ngrid, self.Ncnnout)
        # x0 = torch.cat((x, z0), dim=2)
        # x0 = F.relu(self.linearIn(x0))
        # outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        # out = self.linearOut(outLSTM)
        # # out = rho/time * batchsize * Ntargetvar
        return out

def calConvSize(lin, kernel, stride, padding=0, dilation=1):
    lout = (lin + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return int(lout)

def calPoolSize(lin, kernel, stride=None, padding=0, dilation=1):
    if stride is None:
        stride = kernel
    lout = (lin + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return int(lout)

def calFinalsize1d(nobs, noutk, ksize, stride, pool):
    nlayer = len(ksize)
    Lout = nobs
    for ii in range(nlayer):
        Lout = calConvSize(lin=Lout, kernel=ksize[ii], stride=stride[ii])
        if pool is not None:
            Lout = calPoolSize(lin=Lout, kernel=pool[ii])
    Ncnnout = int(Lout * noutk)  # total CNN feature number after convolution
    return Ncnnout


import numpy as np
import torch
import classDB
import funDB


import imp
imp.reload(classDB)
imp.reload(funDB)


# random seed
rSeed = 0
torch.manual_seed(rSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(rSeed)
np.random.seed(rSeed)

# load data
rootDB = '/mnt/sdc/rnnSMAP/Database_SMAPgrid/Daily_L3/'
subsetName = 'CONUSv4f4'
syr = 2015
eyr = 2016
yrLst = np.arange(syr, eyr+1)

varTuple = ('varLst_soilM', 'varConstLst_Noah')
dataset = classDB.Dataset(rootDB, subsetName, yrLst,
                          var=varTuple, targetName='SMAP_AM')
dataset.readInput(loadNorm=True)
dataset.readTarget(loadNorm=True)

x = dataset.normInput
y = dataset.normTarget
ngrid, nt, nx = x.shape
ny = 1
y = np.reshape(y, [ngrid, nt, ny])


# training parameter
nBatch = 10
nh = 8
rho = 5

# random a piece of time series
xTrain = torch.zeros([rho, nBatch, nx], requires_grad=True)
yTrain = torch.zeros([rho, nBatch, ny], requires_grad=False)
iGrid = np.random.randint(0, ngrid, [nBatch])
iT = np.random.randint(0, nt-rho, [nBatch])

for k in range(nBatch):
    temp = x[iGrid[k]:iGrid[k]+1, np.arange(iT[k], iT[k]+rho), :]
    xTrain[:, k:k+1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    temp = y[iGrid[k]:iGrid[k]+1, np.arange(iT[k], iT[k]+rho)]
    yTrain[:, k:k+1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))

if torch.cuda.is_available():
    xTrain = xTrain.cuda()
    yTrain = yTrain.cuda()


class LSTMModel(torch.nn.Module):
    def __init__(self, nx, ny, nh, nLayer=1):
        super(LSTMModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nh = nh
        self.nLayer = nLayer
        self.lstm = torch.nn.LSTM(nx, nh, nLayer, dropout=0.5)
        self.linear = torch.nn.Linear(nh, ny)

    def forward(self, x):
        if torch.cuda.is_available():
            h0 = torch.zeros(self.nLayer, x.size(
                0), self.nh, requires_grad=True).cuda()
        else:
            h0 = torch.zeros(self.nLayer, x.size(
                0), self.nh, requires_grad=True)

        # Initialize cell state
        if torch.cuda.is_available():
            c0 = torch.zeros(self.nLayer, x.size(
                0), self.nh, requires_grad=True).cuda()
        else:
            c0 = torch.zeros(self.nLayer, x.size(
                0), self.nh, requires_grad=True)
        out, (hn, cn) = self.lstm(x)
        out = self.linear(out)
        return out


model = LSTMModel(nx, ny, nh)
if torch.cuda.is_available():
    model = model.cuda()

crit = torch.nn.MSELoss()
if torch.cuda.is_available():
    crit = crit.cuda()

optim = torch.optim.Adadelta(model.parameters())
# construct model before optim will automatically make it cuda


yP = model(xTrain)
# loc0 = yTrain != yTrain
# loc1 = yTrain == yTrain
# yT=torch.empty(rho,nBatch,1).cuda()
# yT[loc0]=yP[loc0]
# yT[loc1]=yTrain[loc1]

optim.zero_grad()
loss = crit(yP, yTrain)
# loss.backward()
# optim.step()

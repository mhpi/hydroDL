import torch
import torch.nn as nn
import torch.nn.functional as F
from hydroDL.model import cnn, rnn
import hydroDL
from hydroDL import dbCsv
import numpy as np

df = hydroDL.data.dbCsv.DataframeCsv(
    rootDB=hydroDL.pathSMAP['DB_L3_NA'],
    subset='CONUSv4f1',
    tRange=[20150401, 20160401])
x = df.getData(
    varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
y = df.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
nx = x.shape[-1]
ny = 2

# subset
ct = 30
ngrid, nt, nx = x.shape
batchSize, rho = [100, 60]
xTensor = torch.zeros([rho, batchSize, x.shape[-1]], requires_grad=False)
yTensor = torch.zeros([rho, batchSize, y.shape[-1]], requires_grad=False)
cTensor = torch.zeros([ct, batchSize, y.shape[-1]], requires_grad=False)

iGrid = np.random.randint(0, ngrid, [batchSize])
iT = np.random.randint(0, nt - rho, [batchSize])
for k in range(batchSize):
    temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]
    xTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    temp = y[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]
    yTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    ctemp = temp[0, 0:ct, 0]
    i0 = np.where(np.isnan(ctemp))[0]
    i1 = np.where(~np.isnan(ctemp))[0]
    if len(i1) > 0:
        ctemp[i0] = np.interp(i0, i1, ctemp[i1])
        cTensor[:, k:k + 1, :] = torch.from_numpy(
            np.swapaxes(temp[:, 0:ct, :], 1, 0))
if torch.cuda.is_available():
    xTensor = xTensor.cuda()
    yTensor = yTensor.cuda()
    cTensor = cTensor.cuda()

xTrain = xTensor
yTrain = yTensor
cTrain = cTensor

# forward
nx = nx
ny = ny
ct = 30
hiddenSize = 64
cnnSize = 32
cp1 = (64, 3, 2)
cp2 = (128, 5, 2)
dr = 0.5
opt = 3

if opt == 1:  # cnn output as initial state of LSTM (h0)
    cnnSize = hiddenSize
elif opt == 2:  # cnn output as additional output of LSTM
    pass
elif opt == 3:  # cnn output as constant input of LSTM
    pass

cOut, f, p = cp1
conv1 = nn.Conv1d(nx + 1, cOut, f).cuda()
pool1 = nn.MaxPool1d(p).cuda()
lTmp = cnn.calCnnSize(ct, f, 0, 1, 1) / p

cIn = cOut
cOut, f, p = cp2
conv2 = nn.Conv1d(cIn, cOut, f).cuda()
pool2 = nn.MaxPool1d(p).cuda()
lTmp = cnn.calCnnSize(lTmp, f, 0, 1, 1) / p

flatLength = int(cOut * lTmp)
fc1 = nn.Linear(flatLength, cnnSize).cuda()
fc2 = nn.Linear(cnnSize, cnnSize).cuda()

if opt == 3:
    linearIn = torch.nn.Linear(nx + cnnSize, hiddenSize).cuda()
else:
    linearIn = torch.nn.Linear(nx, hiddenSize).cuda()
lstm = rnn.CudnnLstm(inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr).cuda()

if opt == 2:
    linearOut = torch.nn.Linear(hiddenSize + cnnSize, ny).cuda()
else:
    linearOut = torch.nn.Linear(hiddenSize, ny).cuda()

# forward
x1 = torch.cat([cTrain, xTrain[0:ct, :, :]], 2).permute(1, 2, 0)
x1 = pool1(F.relu(conv1(x1)))
x1 = pool2(F.relu(conv2(x1)))
x1 = x1.view(-1, flatLength)
x1 = F.relu(fc1(x1))
x1 = fc2(x1)

x2 = xTrain[ct:, :, :]
if opt == 1:
    x2 = F.relu(linearIn(x2))
    x2, (hn, cn) = lstm(x2, hx=x1[None, :, :])
    x2 = linearOut(x2)
elif opt == 2:
    x1 = x1[None, :, :].repeat(x2.shape[0], 1, 1)
    x2 = F.relu(linearIn(x2))
    x2, (hn, cn) = lstm(x2)
    x2 = linearOut(torch.cat([x2, x1], 2))
elif opt == 3:
    x1 = x1[None, :, :].repeat(x2.shape[0], 1, 1)
    x2 = torch.cat([x2, x1], 2)
    x2 = F.relu(linearIn(x2))
    x2, (hn, cn) = lstm(x2)
    x2 = linearOut(x2)

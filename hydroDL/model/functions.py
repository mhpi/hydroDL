import numpy as np
import torch
import torch.nn.functional as F
import os


def select_subset(
    x, iGrid, iT, rho, c=None, tupleOut=False, LCopt=False, bufftime=0, **kwargs
):
    """
    Selects a subset based on the grid given
    :param x:
    :param iGrid:
    :param iT:
    :param rho:
    :param c:
    :param tupleOut:
    :param LCopt:
    :param bufftime:
    :return:
    """
    nx = x.shape[-1]
    nt = x.shape[1]
    if x.shape[0] == len(iGrid):  # hack
        iGrid = np.arange(0, len(iGrid))  # hack
    if nt <= rho:
        iT.fill(0)
    batchSize = iGrid.shape[0]
    if iT is not None:
        # batchSize = iGrid.shape[0]
        xTensor = torch.zeros([rho + bufftime, batchSize, nx], requires_grad=False)
        for k in range(batchSize):
            temp = x[
                iGrid[k] : iGrid[k] + 1, np.arange(iT[k] - bufftime, iT[k] + rho), :
            ]
            xTensor[:, k : k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    else:
        if LCopt is True:
            # used for local calibration kernel: FDC, SMAP...
            if len(x.shape) == 2:
                # Used for local calibration kernel as FDC
                # x = Ngrid * Ntime
                xTensor = torch.from_numpy(x[iGrid, :]).float()
            elif len(x.shape) == 3:
                # used for LC-SMAP x=Ngrid*Ntime*Nvar
                xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 2)).float()
        else:
            # Used for rho equal to the whole length of time series
            xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 0)).float()
            rho = xTensor.shape[0]
    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(
            np.reshape(c[iGrid, :], [batchSize, 1, nc]), rho + bufftime, axis=1
        )
        cTensor = torch.from_numpy(np.swapaxes(temp, 1, 0)).float()
        if tupleOut:
            if torch.cuda.is_available():
                xTensor = xTensor.cuda()
                cTensor = cTensor.cuda()
            out = (xTensor, cTensor)
        else:
            out = torch.cat((xTensor, cTensor), 2)
    else:
        out = xTensor
    if torch.cuda.is_available() and type(out) is not tuple:
        out = out.cuda()
    return out


def random_index(ngrid, nt, dimSubset, bufftime=0):
    """
    Finds a random place to start inside of the grids
    :param: ngrid: the number of grids
    :param: nt: the number of tiles
    :param: dimSubset: the dimensions of the subset
    :param: bufftime: a buffer
    :returns: the index of the grid and the index of the tile
    """
    batchSize, rho = dimSubset
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0 + bufftime, nt - rho, [batchSize])
    return iGrid, iT


def save_model(outFolder, model, epoch, modelName="model"):
    modelFile = os.path.join(outFolder, modelName + "_Ep" + str(epoch) + ".pt")
    torch.save(model, modelFile)


def load_model(outFolder, epoch, modelName="model"):
    modelFile = os.path.join(outFolder, modelName + "_Ep" + str(epoch) + ".pt")
    model = torch.load(modelFile)
    return model


def invalid_load_func(**kwargs):
    """
    Is called if there is no load_data function defined
    i.e. your model isn't mapped to a load data function
    :param kwargs: keyword arguments
    """
    log.error("no load_data function defined")
    raise Exception("no load_data function defined")

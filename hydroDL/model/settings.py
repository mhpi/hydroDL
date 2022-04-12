"""A file that creates the settings dictionary"""
import torch
import numpy as np
from hydroDL.model import rnn


def make_train_settings(
    model, x, y, c, nEpoch, miniBatch, bufftime,
):
    """
    A function that formats x, and creates the settings dictionary
    :param ModelWrapper: the model
    :param x: input (if a tuple or a list, contains z [additional input])
    :param y: target
    :param c: constant input
    :param num_epochs:
    :param mini_batch:
    :param bufftime:
    :return:
    """
    batchSize, rho = miniBatch
    if type(x) is tuple or type(x) is list:
        x, z = x
    else:
        z = None
    ngrid, nt, nx = x.shape
    if c is not None:
        nx = nx + c.shape[-1]
    if batchSize >= ngrid:
        """batchsize larger than total grids"""
        batchSize = ngrid
    nIterEp = int(
        np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid / (nt - bufftime)))
    )
    if hasattr(model, "ctRm"):
        if model.ctRm is True:
            nIterEp = int(
                np.ceil(
                    np.log(0.01)
                    / np.log(1 - batchSize * (rho - model.ct) / ngrid / (nt - bufftime))
                )
            )
    inputs = {"x": x, "y": y, "z": z, "c": c}
    settings = {
        "ngrid": ngrid,
        "rho": rho,
        "batchSize": batchSize,
        "nt": nt,
        "nx": nx,
        "bufftime": bufftime,
        "nIterEp": nIterEp,
    }
    return inputs, settings


def make_test_settings(model, x, c, batchSize, doMC, outModel):
    if type(x) is tuple or type(x) is list:
        x, z = x
        rnn.CudnnLstmModel
        if type(model) == [rnn.CudnnLstmModel]:
            # For Cudnn, only one input. First concat inputs and obs
            x = np.concatenate([x, z], axis=2)
            z = None
    else:
        z = None
    ngrid, nt, nx = x.shape
    if c is not None:
        nc = c.shape[-1]
    # if type(model) in [physical.CudnnInv_HBVModel]:
    #     ny = 1  # streamflow
    # else:
    #     ny = model.ny
    ny = model.ny
    if batchSize is None:
        batchSize = ngrid
    if hasattr(model, "ctRm"):
        if model.ctRm is True:
            nt = nt - model.ct
    # yP = torch.zeros([nt, ngrid, ny])
    iS = np.arange(0, ngrid, batchSize)
    iE = np.append(iS[1:], ngrid)
    inputs = {"x": x, "z": z, "c": c}
    settings = {
        "batchSize": batchSize,
        "nc": nc,
        "ngrid": ngrid,
        "nt": nt,
        "ny": ny,
        "iS": iS,
        "iE": iE,
        "doMC": doMC,
        "outModel": outModel,
    }
    return inputs, settings

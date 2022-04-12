"""A file that contains training functions requiring z"""
import torch
import numpy as np
from hydroDL.model import rnn


def z_test_1D_swap(inputs, settings, i):
    zTest = inputs["z"]
    if len(inputs["z"].shape) == 2:
        # Used for local calibration kernel as FDC
        zTest = torch.from_numpy(
            inputs["z"][settings["iS"][i] : settings["iE"][i], :]
        ).float()
    elif len(inputs["z"].shape) == 3:
        # used for LC-SMAP
        zTest = torch.from_numpy(
            np.swapaxes(inputs["z"][settings["iS"][i] : settings["iE"][i], :, :], 1, 2)
        ).float()
    return zTest


def z_test_default_swap(inputs, settings, i):
    zTemp = inputs["z"][settings["iS"][i] : settings["iE"][i], :, :]
    # if type(model) in [rnn.CudnnInvLstmModel]: # Test SMAP Inv with attributes
    #     cInv = np.repeat(
    #         np.reshape(c[iS[i]:iE[i], :], [iE[i] - iS[i], 1, nc]), zTemp.shape[1], axis=1)
    #     zTemp = np.concatenate([zTemp, cInv], 2)
    zTest = torch.from_numpy(np.swapaxes(zTemp, 1, 0)).float()
    return zTest


switcher = {
    "CNN1dLCmodel": z_test_1D_swap,
    "CNN1dLCInmodel": z_test_1D_swap,
}


def get_name(model):
    name = ""
    if type(model) in [rnn.CNN1dLCmodel]:
        name = "CNN1dLCmodel"
    elif type(model) in [rnn.CNN1dLCInmodel]:
        name = "CNN1dLCInmodel"
    else:
        name = "default"
    return name


def load_data(inputs, settings, i):
    """
    A testing function for models ANN and LSTM
    :param model: the model you want to train
    :param kwargs: a keyword dictionary for any other passed params
    :return: an array of results. If doMC is true, results are [yp
    """
    data = {}
    name = get_name(model)
    z_swap = switcher.get(name, z_test_default_swap)
    zTest = z_swap(inputs, settings, i)
    if torch.cuda.is_available():
        zTest = zTest.cuda()
    data["x"] = inputs["xTest"]
    data["z"] = zTest
    return data

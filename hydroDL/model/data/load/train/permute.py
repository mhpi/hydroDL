"""A file that contains training functions requiring permutation"""
from hydroDL.model import random_index, select_subset


def load_data(inputs, settings):
    """

    :param model:
    :param x:
    :param y:
    :param z:
    :param c:
    :param settings:
    :return:
    """
    data = {}
    iGrid, iT = random_index(
        settings["ngrid"], settings["nt"], [settings["batchSize"], settings["rho"]]
    )
    xTrain = select_subset(inputs["x"], iGrid, iT, settings["rho"], c=inputs["c"])
    data["x"] = xTrain.permute(1, 2, 0)
    yTrain = select_subset(inputs["y"], iGrid, iT, settings["rho"])
    data["yTrain"] = yTrain.permute(1, 2, 0)[:, :, int(settings["rho"] / 2) :]
    return data

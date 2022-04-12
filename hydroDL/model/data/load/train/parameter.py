"""A file that contains training functions for R2P code"""
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
    # yP = rho/time * Batchsize * Ntraget_var
    iGrid, iT = random_index(
        settings["ngrid"], settings["nt"], [settings["batchSize"], settings["rho"]]
    )
    data["x"] = select_subset(
        inputs["x"], iGrid, iT, settings["rho"], c=inputs["c"], tupleOut=True
    )
    data["yTrain"] = select_subset(inputs["y"], iGrid, iT, settings["rho"])
    return data

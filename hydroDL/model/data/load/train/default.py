"""A file that contains training functions requiring subsets"""
from hydroDL.model import random_index, select_subset


def load_data(inputs, settings):
    """

    :param inputs:
    :param settings:
    :return:
    """
    data = {}
    iGrid, iT = random_index(
        settings["ngrid"], settings["nt"], [settings["batchSize"], settings["rho"]]
    )
    data["x"] = select_subset(inputs["x"], iGrid, iT, settings["rho"], c=inputs["c"])
    data["yTrain"] = select_subset(inputs["y"], iGrid, iT, settings["rho"])
    return data

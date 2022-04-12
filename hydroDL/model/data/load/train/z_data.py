"""A class to hold functions for training that need z data"""
from hydroDL.model import random_index, select_subset


def x_subset_bufftime(**kwargs):
    return select_subset(
        kwargs["x"],
        kwargs["iGrid"],
        kwargs["iT"],
        kwargs["rho"],
        bufftime=kwargs["bufftime"],
    )


def x_subset_default(**kwargs):
    return select_subset(
        kwargs["x"], kwargs["iGrid"], kwargs["iT"], kwargs["rho"], c=kwargs["c"]
    )


def z_subset_no_c(**kwargs):
    return select_subset(kwargs["z"], kwargs["iGrid"], iT=None, rho=None, LCopt=True)


def z_subset_with_c(**kwargs):
    return select_subset(
        kwargs["z"], kwargs["iGrid"], iT=None, rho=None, LCopt=False, c=kwargs["c"]
    )


def z_subset_all(**kwargs):
    return select_subset(
        kwargs["z"], kwargs["iGrid"], iT=kwargs["iT"], rho=kwargs["rho"], c=kwargs["c"]
    )


def z_subset_default(**kwargs):
    return select_subset(
        kwargs["z"], kwargs["iGrid"], iT=kwargs["iT"], rho=kwargs["rho"]
    )


z_switcher = {
    "CNN1dLCmodel": z_subset_no_c,
    "CNN1dLCInmodel": z_subset_no_c,
    "CudnnInvLstmModel": z_subset_with_c,
    "CudnnInv_HBVModel": z_subset_all,
}

x_switcher = {
    "CudnnInv_HBVModel": x_subset_bufftime,
}


def load_data(inputs, settings):
    """

    :param inputs:
    :param settings:
    :return:
    """
    data = {}
    x_train_function = x_switcher.get(settings["name"], x_subset_default)
    z_train_function = z_switcher.get(settings["name"], z_subset_default)
    iGrid, iT = random_index(
        settings["ngrid"],
        settings["nt"],
        [settings["batchSize"], settings["rho"]],
        bufftime=settings["bufftime"],
    )
    data["x"] = x_train_function(
        x=inputs["x"],
        iGrid=iGrid,
        iT=iT,
        rho=settings["rho"],
        c=inputs["c"],
        bufftime=settings["bufftime"],
    )
    data["yTrain"] = select_subset(inputs["y"], iGrid, iT, settings["rho"])
    data["z"] = z_train_function(
        z=inputs["z"], iGrid=iGrid, iT=iT, rho=settings["rho"], c=inputs["c"]
    )
    return data


def load_data_inv(inputs, settings):
    """
    :param inputs:
    :param settings:
    :return:
    """
    data = {}
    iGrid, iT = random_index(
        settings["ngrid"],
        settings["nt"],
        [settings["batchSize"], settings["rho"]],
        bufftime=settings["bufftime"],
    )
    data["x"] = x_subset_bufftime(
        x=inputs["x"],
        iGrid=iGrid,
        iT=iT,
        rho=settings["rho"],
        c=inputs["c"],
        bufftime=settings["bufftime"],
    )
    data["yTrain"] = select_subset(inputs["y"], iGrid, iT, settings["rho"])
    data["z"] = z_subset_all(
        z=inputs["z"], iGrid=iGrid, iT=iT, rho=settings["rho"], c=inputs["c"]
    )
    return data

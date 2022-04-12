import numpy as np
import torch
import time
import os
import hydroDL
from hydroDL.model import cnn, rnn
import pandas as pd
import hydroDL.core.logger as logger
from hydroDL.model.wrappers.ModelWrapper import ModelWrapper
from hydroDL.model.settings import make_test_settings
from hydroDL.model.data.load.test.default import load_data
from hydroDL.model.data.post_process.default import post_process

log = logger.get_logger("model.test_.test")


def testModel(
    model,
    x,
    c,
    *,
    load_data=load_data,
    post_process=post_process,
    batchSize=None,
    filePathLst=None,
    doMC=False,
    outModel=None,
    savePath=None,
):
    # outModel, savePath: only for R2P-hymod model, for other models always set None
    # deal with file name to save
    if filePathLst is None:
        filePathLst = ["out" + str(x) for x in range(settings["ny"])]
    fLst = list()
    for filePath in filePathLst:
        if os.path.exists(filePath):
            os.remove(filePath)
        f = open(filePath, "a")
        fLst.append(f)
    inputs, settings = make_test_settings(model, x, c, batchSize, doMC, outModel)
    _model_ = ModelWrapper(model)
    _model_.model.train(mode=False)
    for i in range(len(settings["iS"])):
        log.debug(f"batch {i}")
        inputs = specify_inputs(inputs, settings, i)
        log.debug(settings["doMC"])
        data = load_data(model, inputs, settings, i)
        results = post_process(_model_(data), inputs, settings)
        yP = results["yP"]
        ySS = results.get("ySS", None)

        # CP-- marks the beginning of problematic merge
        yOut = yP.detach().cpu().numpy().swapaxes(0, 1)
        if settings["doMC"] is not False:
            yOutMC = ySS.swapaxes(0, 1)

        # save output
        for k in range(settings["ny"]):
            f = fLst[k]
            pd.DataFrame(yOut[:, :, k]).to_csv(f, header=False, index=False)
        if settings["doMC"] is not False:
            for k in range(settings["ny"]):
                f = fLst[settings["ny"] + k]
                pd.DataFrame(yOutMC[:, :, k]).to_csv(f, header=False, index=False)
        model.zero_grad()
        torch.cuda.empty_cache()

    for f in fLst:
        f.close()

    if settings["batchSize"] == settings["ngrid"]:
        # For Wenping's work to calculate loss of testing data
        # Only valid for testing without using minibatches
        yOut = torch.from_numpy(yOut)
        if type(model) in [rnn.CudnnLstmModel_R2P]:
            results["Parameters_R2P"] = torch.from_numpy(results["Parameters_R2P"])
            if outModel is None:
                return yOut, results["Parameters_R2P"]
            else:
                return results["q"], results["evap"], results["Parameters_R2P"]
        else:
            return yOut


def specify_inputs(inputs, settings, i):
    """
    Pulls x and c data based on i and the settings
    :return:
    """
    _inputs_ = inputs
    _inputs_["xTemp"] = inputs["x"][settings["iS"][i] : settings["iE"][i], :, :]
    if _inputs_["c"] is not None:
        _inputs_["cTemp"] = np.repeat(
            np.reshape(
                inputs["c"][settings["iS"][i] : settings["iE"][i], :],
                [settings["iE"][i] - settings["iS"][i], 1, settings["nc"]],
            ),
            settings["nt"],
            axis=1,
        )
        _inputs_["xTest"] = torch.from_numpy(
            np.swapaxes(np.concatenate([_inputs_["xTemp"], _inputs_["cTemp"]], 2), 1, 0)
        ).float()
    else:
        _inputs_["xTest"] = torch.from_numpy(
            np.swapaxes(_inputs_["xTemp"], 1, 0)
        ).float()
    if torch.cuda.is_available():
        _inputs_["xTest"] = _inputs_["xTest"].cuda()
    return _inputs_

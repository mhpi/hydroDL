import numpy as np
import torch
import time
import os
from hydroDL.model.settings import make_train_settings
from hydroDL.model.data.load.train.default import load_data
from hydroDL.model import cnn
import pandas as pd
from time import sleep
from tqdm import trange
import hydroDL.core.logger as logger
from hydroDL.model import crit
from hydroDL.model.wrappers.ModelWrapper import ModelWrapper

log = logger.get_logger("model.train_.train")


def trainModel(
    model,
    x,
    y,
    c,
    lossFun,
    nEpoch=500,
    load_data=load_data,
    miniBatch=[100, 30],
    saveEpoch=100,
    saveFolder=None,
    mode="seq2seq",
    bufftime=0,
):
    optim = torch.optim.Adadelta(model.parameters())
    if saveFolder is not None:
        runFile = os.path.join(saveFolder, "run.csv")
        rf = open(runFile, "w+")
    inputs, settings = make_train_settings(model, x, y, c, nEpoch, miniBatch, bufftime,)
    if torch.cuda.is_available():
        lossFun = lossFun.cuda()
    """Explanations of the uniform training interface and ModelWrapper:
        load_data creates dataDict and send it to the model, and it should collaborate with your model. 
        New-style models (model.is_legacy=False) should accept a dict and output a dict
        For legacy-style models (either model.is_legacy=True or does not have .is_legacy), the ModelWrapper will
           accept a dict, extract the correct fields (normally ["x"]) and feed into your model.
        
        You models that accept a Tensor and output a Tensor should work just fine without doing anything as it is handled by default and ModelWrapper.
        This convenience covers the case where constant input "c" is extracted and put in along with ["x"] in load_data
        This means most of the legacy models should train without any changes.
        If it is more complex, what you need to make sure: 
        (i) (Best) for new-style models, load_data assembles the right fields into a dict for your model. 
        (ii) for legacy-style models that accept Tensors, ModelWrapper.pre_process sends in the right items to your model 
             (by default ["x"]) and load_data prepares the right fields (by default ["x"] and ["yTrain"]) for a minibatch. 
             If load_data packs in x,z, then the model can accept (x,z)
        Quick peeks as examples:
        In make_settings we have: if type(x) is tuple, x, z = x; inputs = {"x": x, "y": y, "z": z, "c": c}
            settings = {"batchSize": batchSize,"nc": nc,"ngrid": ngrid,"nt": nt,"ny": ny,"iS": iS,"iE": iE,"doMC": doMC,"outModel": outModel}
        In default load_data: data["x"] = select_subset(inputs["x"], iGrid, iT, settings["rho"], c=inputs["c"])
                              data["yTrain"] = select_subset(inputs["y"], iGrid, iT, settings["rho"])        
        In train/z_data.py/load_data_inv: data["z"] = z_subset_all(z=inputs["z"], iGrid=iGrid, iT=iT, rho=settings["rho"], c=inputs["c"])                      
    """
    _model_ = ModelWrapper(model)  # wraps up so the model can accept a dict
    with trange(1, nEpoch + 1) as pbar:
        for iEpoch in pbar:
            pbar.set_description(f"Training {_model_.model.name}")
            lossEp = 0
            t0 = time.time()
            for iIter in range(0, settings["nIterEp"]):
                _model_.zero_grad()  # this should work even for the wrapper.
                dataDict, iGrid = load_data(inputs, settings)  #
                results = _model_(dataDict)
                if type(lossFun) in [crit.NSELossBatch, crit.NSESqrtLossBatch]:
                    loss = lossFun(results["yP"][bufftime:, :, :], dataDict["yTrain"], igrid=iGrid)
                else:
                    loss = lossFun(results["yP"][bufftime:, :, :], dataDict["yTrain"])
                    # Additional handling code if needed

                loss.backward()
                optim.step()
                lossEp = lossEp + loss.item()

            lossEp = lossEp / settings["nIterEp"]
            loss_str = "Epoch {} Loss {:.3f} time {:.2f}".format(
                iEpoch, lossEp, time.time() - t0
            )
            log.debug(loss_str)
            pbar.set_postfix(loss=lossEp)
            sleep(0.1)
            if saveFolder is not None:
                rf.write(loss_str + "\n")
                if iEpoch % saveEpoch == 0:
                    # save model
                    modelFile = os.path.join(
                        saveFolder, "model_Ep" + str(iEpoch) + ".pt"
                    )
                    torch.save(_model_.model, modelFile)
    if saveFolder is not None:
        rf.close()
    # return model

import os
import rnnSMAP
import numpy as np
import imp
from rnnSMAP import kPath

imp.reload(rnnSMAP)
rnnSMAP.reload()

opt = rnnSMAP.classLSTM.optLSTM(
    out='CONUSv4f1_pytorch', 
    rootDB=rnnSMAP.kPath['DBSMAP_L3_NA'], 
    rootOut=rnnSMAP.kPath['OutSMAP_L3_NA'], 
    syr=2015, eyr=2015, 
    var='varLst_soilM', varC='varConstLst_Noah',
    train='CONUSv4f1'
)

rnnSMAP.funLSTM.trainLSTM(opt)
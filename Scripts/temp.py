
import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
import subprocess

opt = rnnSMAP.classLSTM.optLSTM()
opt['sigma']=1
runTrainLSTM.runCmdLine(opt=opt,cudaID=0,screenName='ttt')
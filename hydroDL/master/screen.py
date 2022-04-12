# run training script in detached screen
# careful! setup python environment before script running

# bash
# source /home/kxf227/anaconda3/bin/activate
# conda activate pytorch

import os
import argparse
from hydroDL import master
from hydroDL.utils import email


def runTrain(masterDict, *, screen="test", cudaID):
    if type(masterDict) is str:
        mFile = masterDict
        masterDict = master.readMasterFile(mFile)
    else:
        mFile = master.writeMasterFile(masterDict)

    codePath = os.path.realpath(__file__)
    if screen is None:
        cmd = "CUDA_VISIBLE_DEVICES={} python {} -F {} -M {}".format(
            cudaID, codePath, "train", mFile
        )
    else:
        cmd = "CUDA_VISIBLE_DEVICES={} screen -dmS {} python {} -F {} -M {}".format(
            cudaID, screen, codePath, "train", mFile
        )

    # if screen is None:
    #     #add some debugs Dapeng
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('-F', dest='func', type=str, default='train')
    #     parser.add_argument('-M', dest='mFile', type=str, default=mFile)
    #     args = parser.parse_args()
    #     if args.func == 'train':
    #         mDict = master.readMasterFile(args.mFile)
    #         master.train(mDict)
    #         # out = mDict['out']
    #         # email.sendEmail(subject='Training Done', text=out)

    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", dest="func", type=str)
    parser.add_argument("-M", dest="mFile", type=str)
    args = parser.parse_args()
    if args.func == "train":
        mDict = master.readMasterFile(args.mFile)
        master.train(mDict)
        # out = mDict['out']
        # email.sendEmail(subject='Training Done', text=out)

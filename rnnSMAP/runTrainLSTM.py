

import rnnSMAP
import argparse
import os
import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

if __name__ == '__main__':
    opt = rnnSMAP.classLSTM.optLSTM()
    parser = opt.toParser()
    args = parser.parse_args()
    opt.fromParser(parser)
    print(opt)
    rnnSMAP.funLSTM.trainLSTM(opt)


def runCmdLine(*, opt, cudaID, screenName='test'):
    argStr = opt.toCmdLine()
    codePath = os.path.realpath(__file__)
    cmd = 'CUDA_VISIBLE_DEVICE='+str(cudaID)+' ' + \
        'screen -dmS '+screenName+' '+'python '+codePath+argStr
    print(cmd)
    os.system(cmd)

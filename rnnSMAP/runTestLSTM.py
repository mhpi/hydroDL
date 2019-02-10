import os
import rnnSMAP
import argparse
import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootDB', dest='rootDB')
    parser.add_argument('--rootOut', dest='rootOut')
    parser.add_argument('--out', dest='out')
    parser.add_argument('--testName', dest='testName')
    parser.add_argument('--syr', dest='syr')
    parser.add_argument('--eyr', dest='eyr')
    args = parser.parse_args()
    print(args)
    rootDB = args.rootDB
    rootOut = args.rootOut
    out = args.out
    testName = args.testName
    syr = int(args.syr)
    eyr = int(args.eyr)
    yrLst = range(syr, eyr+1)

    ds = rnnSMAP.classDB.DatasetPost(
        rootDB=rootDB, subsetName=testName, yrLst=yrLst)
    ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')


def runCmdLine(*, rootDB, rootOut, out, testName, yrLst,
               cudaID, screenName='test'):
    codePath = os.path.realpath(__file__)
    cmd = 'CUDA_VISIBLE_DEVICES='+str(cudaID)+' ' + \
        'screen -dmS '+screenName+' '+'python ' + codePath + \
        ' --rootDB '+str(rootDB) + \
        ' --rootOut '+str(rootOut) + \
        ' --out '+str(out) + \
        ' --testName '+str(testName) + \
        ' --syr '+str(yrLst[0]) + \
        ' --eyr '+str(yrLst[-1])
    print(cmd)
    os.system(cmd)

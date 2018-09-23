import os
import rnnSMAP
import argparse
import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--huc', dest='trainHuc')
    args = parser.parse_args()
    print(args)
    j = int(args.trainHuc)
    print(type(j))
    rootDB = rnnSMAP.kPath['DB_L3_NA']
    rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
    for i in range(0, 18):
        trainName = 'hucn1_'+str(j+1).zfill(2)+'_v2f1'
        testName = 'hucn1_'+str(i+1).zfill(2)+'_v2f1'
        for k in range(1, 2):
            if k == 0:
                out = trainName+'_y15_soilM'
            elif k == 1:
                out = trainName+'_y15_Forcing'
            ds1 = rnnSMAP.classDB.DatasetPost(
                rootDB=rootDB, subsetName=testName, yrLst=[2015])
            ds1.readPred(rootOut=rootOut, out=out, drMC=100,
                         field='LSTM')
            ds2 = rnnSMAP.classDB.DatasetPost(
                rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
            ds2.readPred(rootOut=rootOut, out=out, drMC=100,
                         field='LSTM')


def runTestCmd(*, trainHuc, cudaID, screenName='test'):
    codePath = os.path.realpath(__file__)
    cmd = 'CUDA_VISIBLE_DEVICES='+str(cudaID)+' ' + \
        'screen -dmS '+screenName+' '+'python ' + \
        codePath+' --huc '+str(trainHuc)
    print(cmd)
    os.system(cmd)

#
import rnnSMAP
import numpy as np
import datetime
from pymongo import MongoClient
import bsonnumpy
import time


rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
var = 'varLst_Forcing'
varC = 'varConstLst_Noah'
dataset = rnnSMAP.classDB.DatasetLSTM(
    rootDB=rootDB, subsetName='CONUS',
    yrLst=[2015],
    var=(var, varC), targetName='SMAP_AM')
# x = dataset.readInput()
y = dataset.readTarget()
crd = dataset.crd

client = MongoClient('localhost', 27017)
db = client['dbSMAP-NA']
col = db['SMAP']

tt = time.time()
docLst = list()
for iT in range(0, 10):
    dt = datetime.datetime.utcfromtimestamp(dataset.time[iT].tolist()/1e9)
    doc = {
        'field': 'SMAP_AM',
        'data': {str(i): y[i, iT] for i in range(0, 20)},
        'date': dt
        # 'lat': crd[0:20, 0].tolist(),
        # 'lon': crd[0:20, 1].tolist()
    }
    docLst.append(doc)
col.insert_many(docLst)
print(time.time()-tt)
# col.delete_many({})
# col.create_index('data')

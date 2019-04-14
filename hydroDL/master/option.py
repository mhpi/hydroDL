import hydroDL
from collections import OrderedDict
from hydroDL.data import dbCsv
import json


def saveOpt(opt, fileName):
    if not fileName.endswith('.json'):
        fileName = fileName + '.json'
    with open(fileName, 'w') as fp:
        json.dump(opt, fp, indent=4)


def loadOpt(fileName):
    if not fileName.endswith('.json'):
        fileName = fileName + '.json'
    with open(fileName, 'r') as fp:
        opt = json.load(fp, object_pairs_hook=OrderedDict)
    return opt


def updateOpt(opt, **kw):
    for key in kw:
        if key in opt:
            try:
                opt[key] = type(opt[key])(kw[key])
            except ValueError:
                print('skiped ' + key + ': wrong type')
        else:
            print('skiped ' + key + ': not in argument dict')
    return opt


def readDataOpt(optData, readX=True, readY=True):
    if eval(optData['name']) is hydroDL.data.dbCsv.DataframeCsv:
        df = hydroDL.data.dbCsv.DataframeCsv(
            rootDB=optData['path'],
            subsetName=optData['subset'],
            tRange=optData['dateRange'])
        if readX is True:
            x = df.getData(
                varT=optData['varT'],
                varC=optData['varC'],
                doNorm=optData['doNorm'][0],
                rmNan=optData['rmNan'][0])
        else:
            x = None
        if readY is True:
            y = df.getData(
                varT=optData['target'],
                doNorm=optData['doNorm'][1],
                rmNan=optData['rmNan'][1])
        else:
            y = None
    return (x, y)

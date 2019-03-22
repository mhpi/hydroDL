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


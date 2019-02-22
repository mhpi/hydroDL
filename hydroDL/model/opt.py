
import collections
import os
import argparse
from . import kPath


class optLSTM(collections.OrderedDict):
    def __init__(self, **kw):
        # dataset
        self['rootDB'] = kPath['DB_L3_Global']
        self['rootOut'] = kPath['Out_L3_Global']
        self['gpu'] = 1
        self['out'] = 'test'
        self['train'] = 'Globalv8f1'
        self['var'] = 'varLst_soilM'
        self['varC'] = 'varConstLst_Noah'
        self['target'] = 'SMAP_AM'
        self['syr'] = 2015
        self['eyr'] = 2016
        self['resume'] = 0
        # model
        self['model'] = 'cudnn'
        self['modelOpt'] = 'tied+relu'
        self['hiddenSize'] = 256
        self['dr'] = 0.5
        self['drMethod'] = 'drX+drH+drW+drC'
        self['rho'] = 30
        self['rhoL'] = 30
        self['rhoP'] = 0
        self['nbatch'] = 100
        self['nEpoch'] = 500
        self['saveEpoch'] = 100
        self['addFlag'] = 0
        self['loss'] = 'mse'
        self['lossPrior'] = 'gauss'
        if kw.keys() is not None:
            for key in kw:
                if key in self:
                    try:
                        self[key] = type(self[key])(kw[key])
                    except ValueError:
                        print('skiped '+key+': wrong type')
                else:
                    print('skiped '+key+': not in argument dict')
        self.checkOpt()

    def checkOpt(self):
        if self['model'] == 'cudnn':
            self['modelOpt'] = 'tied+relu'
            self['drMethod'] = 'drW'
        if self['loss'] == 'mse':
            self['lossPrior'] = 'gauss'

    def toParser(self):
        parser = argparse.ArgumentParser()
        for key, value in self.items():
            parser.add_argument('--'+key, dest=key,
                                default=value, type=type(value))
        return parser

    def toCmdLine(self):
        cmdStr = ''
        for key, value in self.items():
            cmdStr = cmdStr+' --'+key+' '+str(value)
        return cmdStr

    def fromParser(self, parser):
        args = parser.parse_args()
        for key, value in self.items():
            self[key] = getattr(args, key)


def loadOptLSTM(outFolder):
    optFile = os.path.join(outFolder, 'opt.txt')
    optTemp = dict()  # type: dict
    with open(optFile, 'r') as ff:
        for line in ff:
            lstTemp = line.strip().split(': ')
            if len(lstTemp) == 1:
                lstTemp = line.strip().split(': ')
                optTemp[lstTemp[0]] = None
            else:
                optTemp[lstTemp[0]] = lstTemp[1]

    opt = optLSTM(**optTemp)
    if opt['rootDB'] is None:
        opt['rootDB'] = kPath.DBSMAP_L3_Global
    if opt['rootOut'] is None:
        opt['rootOut'] = kPath.OutSMAP_L3_Global
    return opt


def saveOptLSTM(outFolder, opt: optLSTM):
    optFile = os.path.join(outFolder, 'opt.txt')
    if os.path.isfile(optFile):
        print('Warning: overwriting existed optFile. Delete manually.')

    with open(optFile, 'w') as ff:
        i = 0
        for key in opt:
            if i != len(opt):
                ff.write(key+': '+str(opt[key])+'\n')
            else:
                ff.write(key+': '+str(opt[key]))
            i = i+1

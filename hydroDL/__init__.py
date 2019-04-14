import os
import socket
import collections
from . import utils
from . import data
from . import model
from . import post

# __all__ = ['classDB', 'funDB', 'classLSTM', 'funLSTM', 'kPath']

print('loading package hydroDL')


def initPath():
    """initial shortcut for some import paths
    """
    hostName = socket.gethostname()
    if hostName == 'smallLinux':
        dirDB = os.path.join(os.path.sep, 'mnt', 'sdc', 'rnnSMAP',
                             'Database_SMAPgrid')
        dirOut = os.path.join(os.path.sep, 'mnt', 'sdb', 'rnnSMAP',
                              'Output_SMAPgrid')
        dirResult = os.path.join(os.path.sep, 'mnt', 'sdb', 'rnnSMAP',
                                 'Result_SMAPgrid')
    pathSMAP = collections.OrderedDict(
        DB_L3_CONUS=os.path.join(dirDB, 'Daily_L3_CONUS'),
        DB_L3_Global=os.path.join(dirDB, 'Daily_L3'),
        DB_L3_NA=os.path.join(dirDB, 'Daily_L3_NA'),
        DB_L4_CONUS=os.path.join(dirDB, 'Daily_L4_CONUS'),
        DB_L4_NA=os.path.join(dirDB, 'Daily_L4_NA'),
        Out_L3_CONUS=os.path.join(dirOut, 'L3_CONUS'),
        Out_L3_Global=os.path.join(dirOut, 'L3_Global'),
        Out_L3_NA=os.path.join(dirOut, 'L3_NA'),
        Out_L4_CONUS=os.path.join(dirOut, 'L4_CONUS'),
        Out_L4_NA=os.path.join(dirOut, 'L4_NA'),
        OutSigma_L3_NA=os.path.join(dirOut, 'L3_NA_sigma'),
        outTest=os.path.join(dirOut, 'Test'),
        dirResult=dirResult)
    return pathSMAP


pathSMAP = initPath()
optHDL = dict(verbose=True)

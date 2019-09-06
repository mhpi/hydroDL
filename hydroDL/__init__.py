import os
import socket
import collections
print('loading package hydroDL')


def initPath():
    """initial shortcut for some import paths
    """
    hostName = socket.gethostname()
    if hostName == 'smallLinux':
        dirDB = os.path.join(os.path.sep, 'mnt', 'sdc', 'rnnSMAP',
                             'Database_SMAPgrid')
        dirOut = os.path.join(os.path.sep, 'mnt', 'sdb', 'rnnSMAP',
                              'Model_SMAPgrid')
        dirResult = os.path.join(os.path.sep, 'mnt', 'sdb', 'rnnSMAP',
                                 'Result_SMAPgrid')
    else:
        dirDB = '/'
        dirOut = '/'
        dirResult = '/'

    pathSMAP = collections.OrderedDict(
        DB_L3_Global=os.path.join(dirDB, 'Daily_L3'),
        DB_L3_NA=os.path.join(dirDB, 'Daily_L3_NA'),
        Out_L3_Global=os.path.join(dirOut, 'L3_Global'),
        Out_L3_NA=os.path.join(dirOut, 'L3_NA'),
        outTest=os.path.join(dirOut, 'Test'),
        dirResult=dirResult)

    pathCamels = collections.OrderedDict(
        DB=os.path.join(os.path.sep, 'mnt', 'sdb', 'Data', 'Camels'),
        Out=os.path.join(os.path.sep, 'mnt', 'sdb', 'rnnStreamflow'))
    return pathSMAP, pathCamels


pathSMAP, pathCamels = initPath()

from . import utils
from . import data
from . import model
from . import post
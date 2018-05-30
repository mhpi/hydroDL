import os
import socket
import collections

__all__ = ['classDB']

print('load rnnSMAP')

#################################################
# initialize data / out path of rnnSMAP


def initPath():
    hostName = socket.gethostname()
    if hostName == 'ce-406chsh11':
        dirDB = os.path.join(
            os.path.sep, 'mnt', 'sdc', 'rnnSMAP', 'Database_SMAPgrid')
        dirOut = os.path.join(
            os.path.sep, 'mnt', 'sdc', 'rnnSMAP', 'Output_SMAPgrid')
    kPath = collections.OrderedDict(
        DBSMAP_L3_CONUS=os.path.join(dirDB, 'Daily_L3_CONUS'),
        DBSMAP_L3_Global=os.path.join(dirDB, 'Daily_L3'),
        DBSMAP_L3_NA=os.path.join(dirDB, 'Daily_L3_NA'),
        DBSMAP_L4_CONUS=os.path.join(dirDB, 'Daily_L4_CONUS'),
        DBSMAP_L4_NA=os.path.join(dirDB, 'Daily_L4_NA'),
        OutSMAP_L3_CONUS=os.path.join(dirOut, 'L3_CONUS'),
        OutSMAP_L3_Global=os.path.join(dirOut, 'L3_Global'),
        OutSMAP_L3_NA=os.path.join(dirOut, 'L3_NA'),
        OutSMAP_L4_CONUS=os.path.join(dirOut, 'L4_CONUS'),
        OutSMAP_L4_NA=os.path.join(dirOut, 'L4_NA'),
    )
    return kPath


kPath = initPath()

#################################################
# import submodules
#################################################
from . import classDB
from . import funDB
from . import classLSTM
from . import funLSTM


def reload():
    import imp
    imp.reload(classDB)
    imp.reload(funDB)
    imp.reload(classLSTM)
    imp.reload(funLSTM)

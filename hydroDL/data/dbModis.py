# ~/anaconda3/envs/envGeoDB/bin/python

import os
import numpy as np
from osgeo import gdal
import psycopg2
import subprocess
from datetime import datetime as dt
import time

dbName = 'gisdb'
schName = 'modis'
varName = 'fpar'

tRg = [dt(2011, 1, 1), dt(2011, 3, 1)]
# hTileRg = [7, 12]
# vTileRg = [4, 6]
hTileRg = [8, 9]
vTileRg = [4, 5]
modisDir = '/mnt/sdb/rawData/MCD15A2H.006/'
fileLst = list()
dateStrLst = [x for x in os.listdir(modisDir) if
              dt.strptime(x, '%Y.%m.%d') >= tRg[0]
              and dt.strptime(x, '%Y.%m.%d') <= tRg[1]]
for dateStr in dateStrLst:
    modisFolder = os.path.join(modisDir, dateStr)
    for fileName in os.listdir(modisFolder):
        if fileName.endswith('.hdf'):
            tileStr = fileName.split('.')[2]
            hTile = int(tileStr[1:3])
            vTile = int(tileStr[4:6])
            if hTile >= hTileRg[0] and hTile <= hTileRg[1] \
                    and vTile >= vTileRg[0] and vTile <= vTileRg[1]:
                fileLst.append(os.path.join(modisDir, dateStr, fileName))


conn = psycopg2.connect(database='gisdb', user='postgres',
                        host="localhost", password='19900323', port=5432)
cursor = conn.cursor()
bTable = False
try:
    cursor = conn.cursor()
    cursor.execute(
        r"SELECT EXISTS(SELECT fpar FROM modis)")
    bTable = True
except psycopg2.Error as errStr:
    print(errStr)

t1 = time.time()
for f in fileLst:
    if bTable is False:
        cmdOpt = '-s 96431 -F -I -R -C -c'
    else:
        cmdOpt = '-s 96431 -F -I -R -C -a'
    cmdFile = 'HDF4_EOS:EOS_GRID:'+f+':MOD_Grid_MOD15A2H:Fpar_500m'
    cmdTab = 'modis.fpar'
    cmdOut = '| psql -d gisdb'
    cmd = ' '.join(['raster2pgsql', cmdOpt, cmdFile, cmdTab, cmdOut])
    subprocess.call(cmd, shell=True)
    if bTable is False:
        # cursor = conn.cursor()
        # cursor.execute(
        #     r"SELECT DropRasterConstraints('modis','fpar', 'rast','enforce_same_alignment_rast')")
        bTable = True
print('load MODIS: ', time.time()-t1)

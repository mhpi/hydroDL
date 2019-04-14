import os
from hydroDL.data import dbCsv
from hydroDL.post import plot, stat
from hydroDL import master

cDir = os.path.dirname(os.path.abspath(__file__))
cDir = r'/home/kxf227/work/GitHUB/pyRnnSMAP/example/'

rootDB = os.path.join(cDir, 'data')
nEpoch = 500
out = os.path.join(cDir, 'output', 'CONUSv4f1')
tRange = [20160401, 20170401]

# load data
df = dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUSv4f1', tRange=tRange)
yt = df.getData(varT='SMAP_AM', doNorm=False, rmNan=False)
yt = yt.squeeze()

yp = master.test(
    out, tRange=[20160401, 20170401], subset='CONUSv4f1', epoch=500)
yp = yp.squeeze()

# calculate stat
statErr = stat.statError(yp, yt)
dataGrid = [statErr['RMSE'], statErr['Corr']]
dataTs = [yp, yt]
t = df.getT()
crd = df.getGeo()
mapNameLst = ['RMSE', 'Correlation']
tsNameLst = ['LSTM', 'SMAP']
colorMap = None
colorTs = None

# plot map and time series
plot.plotTsMap(
    dataGrid,
    dataTs,
    crd,
    t,
    colorMap=colorMap,
    mapNameLst=mapNameLst,
    tsNameLst=tsNameLst)

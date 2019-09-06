import os
from hydroDL.data import dbCsv
from hydroDL.post import plot, stat
from hydroDL import master

cDir = os.path.dirname(os.path.abspath(__file__))

out = os.path.join(cDir, 'output', 'CONUSv4f1')
rootDB = os.path.join(cDir, 'data')
nEpoch = 100
tRange = [20160401, 20170401]

# load data
df, yp, yt = master.test(
    out, tRange=[20160401, 20170401], subset='CONUSv4f1', epoch=100, reTest=True)
yp = yp.squeeze()
yt = yt.squeeze()

# calculate stat
statErr = stat.statError(yp, yt)
dataGrid = [statErr['RMSE'], statErr['Corr']]
dataTs = [yp, yt]
t = df.getT()
crd = df.getGeo()
mapNameLst = ['RMSE', 'Correlation']
tsNameLst = ['LSTM', 'SMAP']

# plot map and time series
plot.plotTsMap(
    dataGrid,
    dataTs,
    lat=crd[0],
    lon=crd[1],
    t=t,
    mapNameLst=mapNameLst,
    tsNameLst=tsNameLst,
    isGrid=True)

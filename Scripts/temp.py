import rnnSMAP
from rnnSMAP import runTrainLSTM
import imp
import numpy as np
from mpl_toolkits.basemap import Basemap, cm
import matplotlib.pyplot as plt
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# Training
opt = rnnSMAP.classLSTM.optLSTM(
    rootDB=rnnSMAP.kPath['DB_L3_NA'],
    rootOut=rnnSMAP.kPath['OutSigma_L3_NA'],
    syr=2015, eyr=2015, varC='varConstLst_Noah',
    dr=0.5, modelOpt='relu',
    model='cudnn', loss='sigma',
    var='varLst_soilM', train='CONUSv4f1'
)
# runTrainLSTM.runCmdLine(opt=opt, cudaID=2, screenName=opt['out'])

##
rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
trainName = 'CONUSv2f1'
testName = 'CONUSv2f1'
out = trainName+'_y15_soilM'
ds = rnnSMAP.classDB.DatasetPost(
    rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
ds.readData(var='SMAP_AM', field='SMAP')
ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')

# crd to grid
crd = ds.crd
ux, indx1, indx2 = np.unique(crd[:, 1], return_index=True, return_inverse=True)
uy, indy1, indy2 = np.unique(crd[:, 0], return_index=True, return_inverse=True)
minDx = np.min(ux[1:]-ux[0:-1])
minDy = np.min(uy[1:]-uy[0:-1])
maxDx = np.max(ux[1:]-ux[0:-1])
maxDy = np.max(uy[1:]-uy[0:-1])
if maxDx > minDx*2 or maxDy > minDy*2:
    raise Exception('skipped coloums or rows')
    print('skipped coloums or rows')
uy = uy[::-1]
ny = len(uy)
indy2 = ny-1-indy2
crdGrid = (uy, ux)
crdGridInd = np.stack((indy2, indx2), axis=1)

# data to grid
data = ds.LSTM
indY = crdGridInd[:, 0]
indX = crdGridInd[:, 1]
ny = len(crdGrid[0])
nx = len(crdGrid[1])
if data.ndim == 2:
    nt = data.shape[1]
    grid = np.full([ny, nx, nt], np.nan)
    grid[indY, indX, :] = data
elif data.ndim == 1:
    grid = np.full([ny, nx], np.nan)
    grid[indY, indX] = data

# draw map
lat = crdGrid[0]
lon = crdGrid[1]
grid = grid[:, :, 0]
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# create polar stereographic Basemap instance.
map = Basemap(llcrnrlat=lat[-1], urcrnrlat=lat[0],
              llcrnrlon=lon[0], urcrnrlon=lon[-1],
              projection='cyl', resolution='c')
map.drawcoastlines()
map.drawstates()
map.drawcountries()
x, y = map(lon, lat)
xx, yy = np.meshgrid(x, y)
cs=map.pcolormesh(xx, yy, grid,cmap=plt.cm.jet)
# cs=map.scatter(xx, yy, c=grid,s=10,cmap=plt.cm.jet,edgecolors=None, linewidth=0)
cbar = map.colorbar(cs,location='bottom',pad="5%")
plt.show()

x=np.array([1,2])
y=np.array([3,4,5])
data=np.array([[1,2],[3,4],[5,6]])
xx,yy=np.meshgrid(x,y)
plt.pcolor(xx,yy,data)
plt.show()

plt.imshow(data)
plt.show()
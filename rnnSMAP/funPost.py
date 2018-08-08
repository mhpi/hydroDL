
# import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import statsmodels.api as sm


def plotBox(data, labelC=None, labelS=None, colorLst='rbkgcmy', title=None):

    nc = len(data)
    fig, axes = plt.subplots(ncols=nc, sharey=True)

    for k in range(0, nc):
        bp = axes[k].boxplot(data[k], patch_artist=True, notch=True)
        for kk in range(0, len(bp['boxes'])):
            plt.setp(bp['boxes'][kk], facecolor=colorLst[kk])

        if labelC is not None:
            axes[k].set_xlabel(labelC[k])
        else:
            axes[k].set_xlabel(str(k))
        axes[k].set_xticks([])

    if labelS is not None:
        axes[-1].legend(bp['boxes'], labelS, loc='upper right')

    fig.suptitle(title)
    plt.show(block=False)


def plotMap(grid, *, lat, lon):
    map = Basemap(llcrnrlat=lat[-1], urcrnrlat=lat[0],
                  llcrnrlon=lon[0], urcrnrlon=lon[-1],
                  projection='cyl', resolution='c')
    map.drawcoastlines()
    map.drawstates()
    map.drawcountries()
    x, y = map(lon, lat)
    xx, yy = np.meshgrid(x, y)
    cs = map.pcolormesh(xx, yy, grid, cmap=plt.cm.jet)
    # cs=map.scatter(xx, yy, c=grid,s=10,cmap=plt.cm.jet,edgecolors=None, linewidth=0)
    cbar = map.colorbar(cs, location='bottom', pad="5%")
    plt.show(block=False)


def regLinear(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    out = sm.OLS(y, X).fit()
    return out

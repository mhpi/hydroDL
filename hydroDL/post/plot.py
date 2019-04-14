import numpy as np
import scipy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from hydroDL import utils

import os
os.environ[
    'PROJ_LIB'] = r'/home/kxf227/anaconda3/pkgs/proj4-5.2.0-he6710b0_1/share/proj/'
from mpl_toolkits import basemap


def plotBoxFig(data,
               label1=None,
               label2=None,
               colorLst='rbkgcmy',
               title=None,
               figsize=(8, 6),
               sharey=True):
    nc = len(data)
    fig, axes = plt.subplots(ncols=nc, sharey=sharey, figsize=figsize)

    for k in range(0, nc):
        ax = axes[k] if nc > 1 else axes
        bp = ax.boxplot(
            data[k], patch_artist=True, notch=True, showfliers=False)
        for kk in range(0, len(bp['boxes'])):
            plt.setp(bp['boxes'][kk], facecolor=colorLst[kk])

        if label1 is not None:
            ax.set_xlabel(label1[k])
        else:
            ax.set_xlabel(str(k))
        ax.set_xticks([])
        # ax.ticklabel_format(axis='y', style='sci')
    if label2 is not None:
        if nc == 1:
            ax.legend(bp['boxes'], label2, loc='upper right')
        else:
            axes[-1].legend(bp['boxes'], label2, loc='upper right')
    if title is not None:
        fig.suptitle(title)
    return fig


def plotTS(t, y, *, ax=None, figsize=(12, 4), cLst='rbkgcmy', legLst=None):
    newFig = False
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots()
        newFig = True

    if type(y) is np.ndarray:
        y = [y]
    for k in range(len(y)):
        tt = t[k] if type(t) is list else t
        yy = y[k]
        legStr = None
        if legLst is not None:
            legStr = legLst[k]
        if True in np.isnan(yy):
            ax.plot(tt, yy, '*', color=cLst[k], label=legStr)
        else:
            ax.plot(tt, yy, color=cLst[k], label=legStr)
    if legLst is not None:
        ax.legend(loc='best')
    if newFig is True:
        return fig, ax
    else:
        return ax


def plotVS(x,
           y,
           *,
           ax=None,
           title=None,
           xlabel=None,
           ylabel=None,
           titleCorr=True,
           plot121=True,
           doRank=False,
           figsize=(8, 6)):
    if doRank is True:
        x = scipy.stats.rankdata(x)
        y = scipy.stats.rankdata(y)
    corr = scipy.stats.pearsonr(x, y)[0]
    pLr = np.polyfit(x, y, 1)
    xLr = np.array([np.min(x), np.max(x)])
    yLr = np.poly1d(pLr)(xLr)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots()
    else:
        fig = None
    if title is not None:
        if titleCorr is True:
            title = title + ' ' + r'$\rho$={:.2f}'.format(corr)
        ax.set_title(title)
    else:
        if titleCorr is True:
            ax.set_title(r'$\rho$=' + '{:.2f}'.format(corr))
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # corr = np.corrcoef(x, y)[0, 1]
    ax.plot(x, y, 'b.')
    ax.plot(xLr, yLr, 'r-')

    if plot121 is True:
        plot121Line(ax)

    return fig, ax


def plot121Line(ax, spec='k-'):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    vmin = np.min([xlim[0], ylim[0]])
    vmax = np.max([xlim[1], ylim[1]])
    ax.plot([vmin, vmax], [vmin, vmax], spec)


def plotMap(grid,
            *,
            crd,
            ax=None,
            lat=None,
            lon=None,
            title=None,
            cRange=None,
            shape=None):
    if lat is None and lon is None:
        lat = crd[0]
        lon = crd[1]
    if cRange is not None:
        vmin = cRange[0]
        vmax = cRange[1]
    else:
        temp = flatData(grid)
        vmin = np.percentile(temp, 5)
        vmax = np.percentile(temp, 95)

    if ax is None:
        fig, ax = plt.figure(figsize=(8, 4))
    mm = basemap.Basemap(
        llcrnrlat=lat[-1],
        urcrnrlat=lat[0],
        llcrnrlon=lon[0],
        urcrnrlon=lon[-1],
        projection='cyl',
        resolution='c',
        ax=ax)
    mm.drawcoastlines()
    # map.drawstates()
    # map.drawcountries()
    x, y = mm(lon, lat)
    xx, yy = np.meshgrid(x, y)
    cs = mm.pcolormesh(xx, yy, grid, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
    if shape is not None:
        crd = np.array(shape.points)
        par = shape.parts
        if len(par) > 1:
            for k in range(0, len(par) - 1):
                x = crd[par[k]:par[k + 1], 0]
                y = crd[par[k]:par[k + 1], 1]
                mm.plot(x, y, color='r', linewidth=3)
        else:
            y = crd[:, 0]
            x = crd[:, 1]
            mm.plot(x, y, color='r', linewidth=3)
    mm.colorbar(cs, location='bottom', pad='5%')
    if title is not None:
        ax.set_title(title)
        if ax is None:
            return fig, ax, mm
        else:
            return mm


def plotTsMap(dataGrid,
              dataTs,
              crd,
              t,
              *,
              colorMap=None,
              mapNameLst=None,
              tsNameLst=None):
    if type(dataGrid) is np.ndarray:
        dataGrid = [dataGrid]
    if type(dataTs) is np.ndarray:
        dataTs = [dataTs]
    nMap = len(dataGrid)
    nTs = len(dataTs)

    fig = plt.figure(figsize=[12, 6])
    gs = gridspec.GridSpec(3, nMap)

    for k in range(nMap):
        ax = fig.add_subplot(gs[0:2, k])
        data = dataGrid[k]
        grid, uy, ux = utils.grid.array2grid(data, crd)
        cRange = None if colorMap is None else colorMap[k]
        title = None if mapNameLst is None else mapNameLst[k]
        plotMap(grid, crd=[uy, ux], ax=ax, cRange=cRange, title=title)
    axTs = fig.add_subplot(gs[2, :])

    def onclick(event):
        lon = event.xdata
        lat = event.ydata
        d = np.sqrt((lon - crd[:, 1])**2 + (lat - crd[:, 0])**2)
        ind = np.argmin(d)
        tsLst = list()
        for k in range(nTs):
            tsLst.append(dataTs[k][ind, :])
        axTs.clear()
        plotTS(t, tsLst, ax=axTs, legLst=tsNameLst)
        plt.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()


def plotCDF(xLst,
            *,
            ax=None,
            title=None,
            legendLst=None,
            figsize=(8, 6),
            ref='121',
            cLst=None,
            xlabel=None,
            ylabel=None,
            showDiff='RMSE'):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots()
    else:
        fig = None

    if cLst is None:
        cmap = plt.cm.jet
        cLst = cmap(np.linspace(0, 1, len(xLst)))

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    xSortLst = list()
    rmseLst = list()
    ksdLst = list()
    for k in range(0, len(xLst)):
        x = xLst[k]
        xSort = flatData(x)
        yRank = np.arange(len(xSort)) / float(len(xSort) - 1)
        xSortLst.append(xSort)
        if legendLst is None:
            legStr = None
        else:
            legStr = legendLst[k]

        if ref is '121':
            yRef = yRank
        elif ref is 'norm':
            yRef = scipy.stats.norm.cdf(xSort, 0, 1)
        rmse = np.sqrt(((xSort - yRef)**2).mean())
        ksd = np.max(np.abs(xSort - yRef))
        rmseLst.append(rmse)
        ksdLst.append(ksd)
        if showDiff is 'RMSE':
            legStr = legStr + ' RMSE=' + '%.3f' % rmse
        elif showDiff is 'KS':
            legStr = legStr + ' KS=' + '%.3f' % ksd
        ax.plot(xSort, yRank, color=cLst[k], label=legStr)

    if ref is '121':
        ax.plot([0, 1], [0, 1], 'k', label='y=x')
    if ref is 'norm':
        xNorm = np.linspace(-5, 5, 1000)
        normCdf = scipy.stats.norm.cdf(xNorm, 0, 1)
        ax.plot(xNorm, normCdf, 'k', label='Gaussian')
    if legendLst is not None:
        ax.legend(loc='best')
    out = {'xSortLst': xSortLst, 'rmseLst': rmseLst, 'ksdLst': ksdLst}
    return fig, ax, out


def flatData(x):
    xArrayTemp = x.flatten()
    xArray = xArrayTemp[~np.isnan(xArrayTemp)]
    xSort = np.sort(xArray)
    return (xSort)


def scaleSigma(s, u, y):
    yNorm = (y - u) / s
    _, sF = scipy.stats.norm.fit(flatData(yNorm))
    return sF


def reCalSigma(s, u, y):
    conf = scipy.special.erf(np.abs(y - u) / s / np.sqrt(2))
    yNorm = (y - u) / s
    return conf, yNorm


def regLinear(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    out = sm.OLS(y, X).fit()
    return out

import numpy as np
import scipy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from hydroDL import utils
import string

import os
# manually add package
# os.environ[
#    'PROJ_LIB'] = r'C:\pythonenvir\pkgs\proj4-5.2.0-ha925a31_1\Library\share'
from mpl_toolkits import basemap


def plotBoxFig(data,
               label1=None,
               label2=None,
               colorLst='rbkgcmy',
               title=None,
               figsize=(8, 6),
               sharey=True,
               legOnly=False):
    nc = len(data)
    fig, axes = plt.subplots(ncols=nc, sharey=sharey, figsize=figsize)

    for k in range(0, nc):
        ax = axes[k] if nc > 1 else axes
        temp = data[k]
        if type(temp) is list:
            for kk in range(len(temp)):
                tt = temp[kk]
                if tt is not None and tt != []:
                    tt = tt[~np.isnan(tt)]
                    temp[kk] = tt
                else:
                    temp[kk] = []
        else:
            temp = temp[~np.isnan(temp)]
        bp = ax.boxplot(temp, patch_artist=True, notch=True, showfliers=False)
        for kk in range(0, len(bp['boxes'])):
            plt.setp(bp['boxes'][kk], facecolor=colorLst[kk])
        if label1 is not None:
            ax.set_xlabel(label1[k])
        else:
            ax.set_xlabel(str(k))
        ax.set_xticks([])
        # ax.ticklabel_format(axis='y', style='sci')
    if label2 is not None:
        ax.legend(bp['boxes'], label2, loc='best')
        if legOnly is True:
            ax.legend(bp['boxes'], label2, bbox_to_anchor=(1, 0.5))
    if title is not None:
        fig.suptitle(title)
    return fig


def plotTS(t,
           y,
           *,
           ax=None,
           tBar=None,
           figsize=(12, 4),
           cLst='rbkgcmy',
           markerLst=None,
           legLst=None,
           title=None,
           linewidth=2):
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
        if markerLst is None:
            if True in np.isnan(yy):
                ax.plot(tt, yy, '*', color=cLst[k], label=legStr)
            else:
                ax.plot(
                    tt, yy, color=cLst[k], label=legStr, linewidth=linewidth)
        else:
            if markerLst[k] is '-':
                ax.plot(
                    tt, yy, color=cLst[k], label=legStr, linewidth=linewidth)
            else:
                ax.plot(
                    tt, yy, color=cLst[k], label=legStr, marker=markerLst[k])
        # ax.set_xlim([np.min(tt), np.max(tt)])
    if tBar is not None:
        ylim = ax.get_ylim()
        tBar = [tBar] if type(tBar) is not list else tBar
        for tt in tBar:
            ax.plot([tt, tt], ylim, '-k')
    if legLst is not None:
        ax.legend(loc='best')
    if title is not None:
        ax.set_title(title)
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


def plotMap(data,
            *,
            ax=None,
            lat=None,
            lon=None,
            title=None,
            cRange=None,
            shape=None,
            pts=None,
            figsize=(8, 4),
            plotColorBar=True):
    if cRange is not None:
        vmin = cRange[0]
        vmax = cRange[1]
    else:
        temp = flatData(data)
        vmin = np.percentile(temp, 5)
        vmax = np.percentile(temp, 95)
    if ax is None:
        fig, ax = plt.figure(figsize=figsize)

    if len(data.squeeze().shape) == 1:
        isGrid = False
    else:
        isGrid = True

    mm = basemap.Basemap(
        llcrnrlat=np.min(lat),
        urcrnrlat=np.max(lat),
        llcrnrlon=np.min(lon),
        urcrnrlon=np.max(lon),
        projection='cyl',
        resolution='c',
        ax=ax)
    mm.drawcoastlines()
    mm.drawstates()
    # map.drawcountries()
    x, y = mm(lon, lat)
    if isGrid is True:
        xx, yy = np.meshgrid(x, y)
        cs = mm.pcolormesh(xx, yy, data, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
        # cs = mm.imshow(
        #     np.flipud(data),
        #     cmap=plt.cm.jet,
        #     vmin=vmin,
        #     vmax=vmax,
        #     extent=[x[0], x[-1], y[0], y[-1]])
    else:
        cs = mm.scatter(
            x, y, c=data, s=30, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)

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
    if pts is not None:
        mm.plot(pts[1], pts[0], 'k*', markersize=4)
        npt = len(pts[0])
        for k in range(npt):
            plt.text(
                pts[1][k],
                pts[0][k],
                string.ascii_uppercase[k],
                fontsize=18)
    if plotColorBar is True:
        mm.colorbar(cs, location='bottom', pad='5%')
    if title is not None:
        ax.set_title(title)
        if ax is None:
            return fig, ax, mm
        else:
            return mm


def plotTsMap(dataMap,
              dataTs,
              *,
              lat,
              lon,
              t,
              dataTs2=None,
              tBar=None,
              mapColor=None,
              tsColor='krbg',
              tsColor2='cmy',
              mapNameLst=None,
              tsNameLst=None,
              tsNameLst2=None,
              figsize=[12, 6],
              isGrid=False,
              multiTS=False,
              linewidth=1):
    if type(dataMap) is np.ndarray:
        dataMap = [dataMap]
    if type(dataTs) is np.ndarray:
        dataTs = [dataTs]
    if dataTs2 is not None:
        if type(dataTs2) is np.ndarray:
            dataTs2 = [dataTs2]
    nMap = len(dataMap)

    # setup axes
    fig = plt.figure(figsize=figsize)
    if multiTS is False:
        nAx = 1
        dataTs = [dataTs]
        if dataTs2 is not None:
            dataTs2 = [dataTs2]
    else:
        nAx = len(dataTs)
    gs = gridspec.GridSpec(3 + nAx, nMap)
    gs.update(wspace=0.025, hspace=0)
    axTsLst = list()
    for k in range(nAx):
        axTs = fig.add_subplot(gs[k + 3, :])
        axTsLst.append(axTs)
    if dataTs2 is not None:
        axTs2Lst = list()
        for axTs in axTsLst:
            axTs2 = axTs.twinx()
            axTs2Lst.append(axTs2)

    # plot maps
    for k in range(nMap):
        ax = fig.add_subplot(gs[0:2, k])
        cRange = None if mapColor is None else mapColor[k]
        title = None if mapNameLst is None else mapNameLst[k]
        data = dataMap[k]
        if isGrid is False:
            plotMap(data, lat=lat, lon=lon, ax=ax, cRange=cRange, title=title)
        else:
            grid, uy, ux = utils.grid.array2grid(data, lat=lat, lon=lon)
            plotMap(grid, lat=uy, lon=ux, ax=ax, cRange=cRange, title=title)

    # plot ts
    def onclick(event):
        xClick = event.xdata
        yClick = event.ydata
        d = np.sqrt((xClick - lon)**2 + (yClick - lat)**2)
        ind = np.argmin(d)
        titleStr = 'pixel %d, lat %.3f, lon %.3f' % (ind, lat[ind], lon[ind])
        for ix in range(nAx):
            tsLst = list()
            for temp in dataTs[ix]:
                tsLst.append(temp[ind, :])
            axTsLst[ix].clear()
            if ix == 0:
                plotTS(
                    t,
                    tsLst,
                    ax=axTsLst[ix],
                    legLst=tsNameLst,
                    title=titleStr,
                    cLst=tsColor,
                    linewidth=linewidth,
                    tBar=tBar)
            else:
                plotTS(
                    t,
                    tsLst,
                    ax=axTsLst[ix],
                    legLst=tsNameLst,
                    cLst=tsColor,
                    linewidth=linewidth,
                    tBar=tBar)

            if dataTs2 is not None:
                tsLst2 = list()
                for temp in dataTs2[ix]:
                    tsLst2.append(temp[ind, :])
                axTs2Lst[ix].clear()
                plotTS(
                    t,
                    tsLst2,
                    ax=axTs2Lst[ix],
                    legLst=tsNameLst2,
                    cLst=tsColor2,
                    lineWidth=linewidth,
                    tBar=tBar)
            if ix != nAx - 1:
                axTsLst[ix].set_xticklabels([])
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

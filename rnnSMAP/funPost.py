
# import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import statsmodels.api as sm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def plotBox(data, labelC=None, labelS=None, colorLst='rbkgcmy', title=None,
            figsize=(8, 6)):
    nc = len(data)
    fig, axes = plt.subplots(ncols=nc, sharey=True, figsize=figsize)

    for k in range(0, nc):
        ax = axes[k] if nc > 1 else axes
        bp = ax.boxplot(data[k], patch_artist=True, notch=True)
        for kk in range(0, len(bp['boxes'])):
            plt.setp(bp['boxes'][kk], facecolor=colorLst[kk])

        if labelC is not None:
            ax.set_xlabel(labelC[k])
        else:
            ax.set_xlabel(str(k))
        ax.set_xticks([])
    if labelS is not None:
        if nc == 1:
            ax.legend(bp['boxes'], labelS, loc='upper right')
        else:
            axes[-1].legend(bp['boxes'], labelS, loc='upper right')
    fig.suptitle(title)
    plt.show(block=False)
    return fig


def plotVS(x, y, *, ax=None, title=None, xlabel=None, ylabel=None,
           titleCorr=True, plot121=True, doRank=False, figsize=(8, 6)):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots()
    else:
        fig = None

    # corr = np.corrcoef(x, y)[0, 1]
    if doRank is True:
        x = scipy.stats.rankdata(x)
        y = scipy.stats.rankdata(y)
    corr = scipy.stats.pearsonr(x, y)[0]
    pLr = np.polyfit(x, y, 1)
    xLr = np.array([np.min(x), np.max(x)])
    yLr = np.poly1d(pLr)(xLr)
    ax.plot(x, y, 'b*')
    ax.plot(xLr, yLr, 'r-')

    if plot121 is True:
        plot121Line(ax)
    if title is not None:
        if titleCorr is True:
            title = title+' '+'Pearson Correlation '+'{:.2f}'.format(corr)
        ax.set_title(title)
    else:
        if titleCorr is True:
            ax.set_title('Pearson Correlation '+'{:.2f}'.format(corr))
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return fig, ax


def plot121Line(ax, spec='k-'):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    vmin = np.min([xlim[0], ylim[0]])
    vmax = np.max([xlim[1], ylim[1]])
    ax.plot([vmin, vmax], [vmin, vmax], spec)


def plotMap(grid, *, crd, lat=None, lon=None, title=None, showFig=True,
            saveFile=None, cRange=None):
    if lat is None and lon is None:
        lat = crd[0]
        lon = crd[1]
    vmin = cRange[0] if cRange is not None else None
    vmax = cRange[1] if cRange is not None else None

    fig = plt.figure(figsize=(8, 4))
    map = Basemap(llcrnrlat=lat[-1], urcrnrlat=lat[0],
                  llcrnrlon=lon[0], urcrnrlon=lon[-1],
                  projection='cyl', resolution='c')
    map.drawcoastlines()
    map.drawstates()
    map.drawcountries()
    x, y = map(lon, lat)
    xx, yy = np.meshgrid(x, y)
    cs = map.pcolormesh(xx, yy, grid, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
    # cs=map.scatter(xx, yy, c=grid,s=10,cmap=plt.cm.jet,edgecolors=None, linewidth=0)
    cbar = map.colorbar(cs, location='bottom', pad="5%")
    if title is not None:
        fig.suptitle(title)
    if showFig is True:
        fig.show(block=False)
    if saveFile is not None:
        fig.savefig(saveFile)
    return fig


def regLinear(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    out = sm.OLS(y, X).fit()
    return out


def plotTwinBox(ax, xdata, ydata, xerror, yerror, facecolor='r',
                edgecolor='None', alpha=0.5):

    # Create list for all the error patches
    errorboxes = []

    # Loop over data points; create box from errors at each point
    for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T):
        rect = Rectangle(
            (x - xe[0], y - ye[0]), xe.sum(), ye.sum())
        errorboxes.append(rect)

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(
        errorboxes, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor)

    # Add collection to axes
    ax.add_collection(pc)

    # Plot errorbars
    artists = ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror,
                          fmt='None', ecolor='k')

    return artists

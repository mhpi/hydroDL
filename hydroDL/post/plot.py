import numpy as np
import scipy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from hydroDL import utils
import string
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

# from mpl_toolkits import basemap


def plotBoxFig(data,
               label1=None,
               label2=None,
               colorLst='rbkgcmywrbkgcmyw',
               title=None,
               figsize=(10, 8),
               sharey=True,
               xticklabel=None,
               axin=None,
               ylim=None,
               ylabel=None,
               widths=0.5,
               ):
    nc = len(data)
    if axin is None:
        fig, axes = plt.subplots(ncols=nc, sharey=sharey, figsize=figsize, constrained_layout=True)
    else:
        axes = axin

    for k in range(0, nc):
        ax = axes[k] if nc > 1 else axes
        temp = data[k]
        if type(temp) is list:
            for kk in range(len(temp)):
                tt = temp[kk]
                if tt is None or (isinstance(tt, np.ndarray) and tt.size == 0):
                    temp[kk] = []
                else:
                    tt = tt[~np.isnan(tt)]
                    temp[kk] = tt
        else:
            temp = temp[~np.isnan(temp)]
        bp = ax.boxplot(temp, patch_artist=True, notch=True, showfliers=False, widths = widths)
        for kk in range(0, len(bp['boxes'])):
            plt.setp(bp['boxes'][kk], facecolor=colorLst[kk])

        if label1 is not None:
            ax.set_xlabel(label1[k])
        else:
            ax.set_xlabel(str(k))
        if xticklabel is None:
            ax.set_xticks([])
        else:
            ax.set_xticks([y+1 for y in range(0,len(data[k]),2)])
            ax.set_xticklabels(xticklabel)
        # ax.ticklabel_format(axis='y', style='sci')
        if ylabel is not None:
            ax.set_ylabel(ylabel[k])
        # yh = np.nanmedian(data[k][0])
        # ax.axhline(yh, xmin=0, xmax=1, color='r',
        #            linestyle='dashed', linewidth=2)
        # yh1 = np.nanmedian(data[k][1])
        # ax.axhline(yh1, xmin=0, xmax=1, color='b',
        #            linestyle='dashed', linewidth=2)
        if ylim is not None:
            ax.set_ylim(ylim)
    if label2 is not None:
        if nc == 1:
            ax.legend(bp['boxes'], label2, loc='lower center', frameon=False, ncol=2)
        else:
            axes[-1].legend(bp['boxes'], label2, loc='lower center', frameon=False, ncol=2, fontsize=12)
    if title is not None:
        # fig.suptitle(title)
        ax.set_title(title)
    if axin is None:
        return fig
    else:
        return ax, bp


def plotBoxF(data,
               label1=None,
               label2=None,
               colorLst='rbkgcmy',
               title=None,
               figsize=(10, 8),
               sharey=True,
               xticklabel=None,
               ylabel=None,
               subtitles=None
               ):
    nc = len(data)
    fig, axes = plt.subplots(nrows=3, ncols=2, sharey=sharey, figsize=figsize, constrained_layout=True)
    axes = axes.flat
    for k in range(0, nc):
        ax = axes[k] if nc > 1 else axes
        # ax = axes[k]
        bp = ax.boxplot(
            data[k], patch_artist=True, notch=True, showfliers=False)
        for kk in range(0, len(bp['boxes'])):
            plt.setp(bp['boxes'][kk], facecolor=colorLst[0])
        if k == 2:
            yrange = ax.get_ylim()
        if k == 3:
            ax.set(ylim=yrange)
        ax.axvline(len(data[k])-3+0.5, ymin=0, ymax=1, color='k',
                   linestyle='dashed', linewidth=1)
        if ylabel[k] not in ['NSE', 'Corr', 'RMSE', 'KGE']:
            ax.axhline(0, xmin=0, xmax=1,color='k',
                       linestyle='dashed', linewidth=1)

        if label1 is not None:
            ax.set_xlabel(label1[k])
        if ylabel is not None:
            ax.set_ylabel(ylabel[k])
        if xticklabel is None:
            ax.set_xticks([])
        else:
            ax.set_xticks([y+1 for y in range(0,len(data[k]))])
            ax.set_xticklabels(xticklabel)
        if subtitles is not None:
            ax.set_title(subtitles[k], loc='left')
        # ax.ticklabel_format(axis='y', style='sci')
    if label2 is not None:
        if nc == 1:
            ax.legend(bp['boxes'], label2, loc='best', frameon=False, ncol=2)
        else:
            axes[-1].legend(bp['boxes'], label2, loc='best', frameon=False, ncol=2, fontsize=12)
    if title is not None:
        fig.suptitle(title)
    return fig

def plotMultiBoxFig(data,
               *,
               axes=None,
               label1=None,
               label2=None,
               colorLst='grbkcmy',
               title=None,
               figsize=(10, 8),
               sharey=True,
               xticklabel=None,
               position=None,
               ylabel=None,
               ylim = None,
               ):
    nc = len(data)
    if axes is None:
        fig, axes = plt.subplots(ncols=nc, sharey=sharey, figsize=figsize, constrained_layout=True)
    nv = len(data[0])
    ndays = len(data[0][1])-1
    for k in range(0, nc):
        ax = axes[k] if nc > 1 else axes
        bp = [None]*nv
        for ii in range(nv):
            bp[ii] = ax.boxplot(
            data[k][ii], patch_artist=True, notch=True, showfliers=False, positions=position[ii], widths=0.2)
            for kk in range(0, len(bp[ii]['boxes'])):
                plt.setp(bp[ii]['boxes'][kk], facecolor=colorLst[ii])

        if label1 is not None:
            ax.set_xlabel(label1[k])
        else:
            ax.set_xlabel(str(k))
        if ylabel is not None:
            ax.set_ylabel(ylabel[k])
        if xticklabel is None:
            ax.set_xticks([])
        else:
            ax.set_xticks([-0.7]+[y for y in range(0,len(data[k][1])+1)])
            # ax.set_xticks([y for y in range(0, len(data[k][1]) + 1)])
            # xtickloc = [0.25, 0.75] + np.arange(1.625, 5, 1.25).tolist() + [5.5, 5.5+0.25*6]
            # ax.set_xticks([y for y in xtickloc])
            ax.set_xticklabels(xticklabel)
        # ax.set_xlim([0.0, 7.75])
        ax.set_xlim([-0.9, ndays + 0.5])
        # ax.set_xlim([-0.5, ndays + 0.5])
        # ax.ticklabel_format(axis='y', style='sci')
        # vlabel = [0.5] + np.arange(1.0, 5, 1.25).tolist() + [4.75+0.25*6, 4.75+0.25*12]
        vlabel = np.arange(-0.5, len(data[k][1]) + 1)
        for xv in vlabel:
            ax.axvline(xv, ymin=0, ymax=1, color='k',
                       linestyle='dashed', linewidth=1)
        yh0 = np.nanmedian(data[k][0][0])
        ax.axhline(yh0, xmin=0, xmax=1, color='grey',
                   linestyle='dashed', linewidth=2)
        yh = np.nanmedian(data[k][0][1])
        ax.axhline(yh, xmin=0, xmax=1, color='r',
                   linestyle='dashed', linewidth=2)
        yh1 = np.nanmedian(data[k][1][0])
        ax.axhline(yh1, xmin=0, xmax=1, color='b',
                   linestyle='dashed', linewidth=2)
        if ylim is not None:
            ax.set_ylim(ylim)
    labelhandle = list()
    for ii in range(nv):
        labelhandle.append(bp[ii]['boxes'][0])
    if label2 is not None:
        if nc == 1:
            ax.legend(labelhandle, label2, loc='lower center', frameon=False, ncol=2)
        else:
            axes[-1].legend(labelhandle, label2, loc='lower center', frameon=False, ncol=1, fontsize=12)
    if title is not None:
        # fig.suptitle(title)
        ax.set_title(title)
    if axes is None:
        return fig
    else:
        return ax, labelhandle


def plotTS(t,
           y,
           *,
           ax=None,
           tBar=None,
           figsize=(12, 4),
           cLst='rbkgcmy',
           markerLst=None,
           linespec=None,
           legLst=None,
           title=None,
           linewidth=2,
           ylabel=None):
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
            if markerLst[k] == '-':
                if linespec is not None:
                    ax.plot(tt, yy, color=cLst[k], label=legStr, linestyle=linespec[k], lw=1.5)
                else:
                    ax.plot(tt, yy, color=cLst[k], label=legStr, lw=1.5)
            else:
                ax.plot(
                    tt, yy, color=cLst[k], label=legStr, marker=markerLst[k])
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        # ax.set_xlim([np.min(tt), np.max(tt)])
    if tBar is not None:
        ylim = ax.get_ylim()
        tBar = [tBar] if type(tBar) is not list else tBar
        for tt in tBar:
            ax.plot([tt, tt], ylim, '-k')

    if legLst is not None:
        ax.legend(loc='upper right', frameon=False)
    if title is not None:
        ax.set_title(title, loc='center')
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

def plotxyVS(x,
           y,
           *,
           ax=None,
           title=None,
           xlabel=None,
           ylabel=None,
           titleCorr=True,
           plot121=True,
           plotReg=False,
           corrType='Pearson',
           figsize=(8, 6),
           markerType = 'ob'):
    if corrType == 'Pearson':
        corr = scipy.stats.pearsonr(x, y)[0]
    elif corrType == 'Spearman':
        corr = scipy.stats.spearmanr(x, y)[0]
    rmse = np.sqrt(np.nanmean((x - y)**2))
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
            title = title + ' ' + r'$\rho$={:.2f}'.format(corr) + ' ' + r'$RMSE$={:.3f}'.format(rmse)
        ax.set_title(title)
    else:
        if titleCorr is True:
            ax.set_title(r'$\rho$=' + '{:.2f}'.format(corr)) + ' ' + r'$RMSE$={:.3f}'.format(rmse)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.plot(x, y, markerType, markerfacecolor='none')
    # ax.set_xlim(min(np.min(x), np.min(y))-0.1, max(np.max(x), np.max(y))+0.1)
    # ax.set_ylim(min(np.min(x), np.min(y))-0.1, max(np.max(x), np.max(y))+0.1)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(x), np.max(x))
    if plotReg is True:
        ax.plot(xLr, yLr, 'r-')
    ax.set_aspect('equal', 'box')
    if plot121 is True:
        plot121Line(ax)
        # xyline = np.linspace(*ax.get_xlim())
        # ax.plot(xyline, xyline)

    return fig, ax

def plot121Line(ax, spec='k-'):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    vmin = np.min([xlim[0], ylim[0]])
    vmax = np.max([xlim[1], ylim[1]])
    ax.plot([vmin, vmax], [vmin, vmax], spec)


def plotMap(data,
            *,
            fig=None,
            ax=None,
            lat=None,
            lon=None,
            title=None,
            cRange=None,
            shape=None,
            pts=None,
            figsize=(8, 4),
            clbar=True,
            cRangeint=False,
            cmap=plt.cm.jet,
            bounding=None,
            prj='cyl'):

    if cRange is not None:
        vmin = cRange[0]
        vmax = cRange[1]
    else:
        temp = flatData(data)
        vmin = np.percentile(temp, 5)
        vmax = np.percentile(temp, 95)
        if cRangeint is True:
            vmin = int(round(vmin))
            vmax = int(round(vmax))
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots()

    if len(data.squeeze().shape) == 1:
        isGrid = False
    else:
        isGrid = True
    if bounding is None:
        bounding = [np.min(lon)-0.5, np.max(lon)+0.5,
                    np.min(lat)-0.5,np.max(lat)+0.5]
        # bounding = [np.min(lat)-0.5, np.max(lat)+0.5,
        #             np.min(lon)-0.5,np.max(lon)+0.5]

    ax.set_extent(bounding, crs=ccrs.Geodetic())
    ax.add_feature(cfeature.OCEAN)  # ocean backgroud
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES, linestyle = ':')
    # ax.add_feature(cfeature.STATES.with_scale('110m'))

    # mm = basemap.Basemap(
    #     llcrnrlat=bounding[0],
    #     urcrnrlat=bounding[1],
    #     llcrnrlon=bounding[2],
    #     urcrnrlon=bounding[3],
    #     projection=prj,
    #     resolution='c',
    #     ax=ax)
    # mm.drawcoastlines()
    # mm.drawstates(linestyle='dashed')
    # mm.drawcountries(linewidth=1.0, linestyle='-.')
    # x, y = mm(lon, lat)
    x, y = lon, lat
    if isGrid is True:
        xx, yy = np.meshgrid(x, y)
        # cs = mm.pcolormesh(xx, yy, data, cmap=cmap, vmin=vmin, vmax=vmax)
        cs = ax.pcolormesh(lon, lat, data, rasterized=True, )

        # cs = mm.imshow(
        #     np.flipud(data),
        #     cmap=plt.cm.jet(np.arange(0, 1, 0.1)),
        #     vmin=vmin,
        #     vmax=vmax,
        #     extent=[x[0], x[-1], y[0], y[-1]])
    else:

        # cs = mm.scatter(
        #     x, y, c=data, s=30, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
        cs = ax.scatter(
            x, y, c=data, s=30, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)


    if shape is not None:
        crd = np.array(shape.points)
        par = shape.parts
        if len(par) > 1:
            for k in range(0, len(par) - 1):
                x = crd[par[k]:par[k + 1], 0]
                y = crd[par[k]:par[k + 1], 1]
                # mm.plot(x, y, color='r', linewidth=3)
                ax.plot(x, y, color='r', linewidth=3)
        else:
            y = crd[:, 0]
            x = crd[:, 1]
            # mm.plot(x, y, color='r', linewidth=3)
            ax.plot(x, y, color='r', linewidth=3)
    if pts is not None:
        # mm.plot(pts[1], pts[0], 'k*', markersize=4)
        ax.plot(pts[1], pts[0], 'k*', markersize=4)
        npt = len(pts[0])
        for k in range(npt):
            plt.text(
                pts[1][k],
                pts[0][k],
                string.ascii_uppercase[k],
                fontsize=18)
    if clbar is True:
        fig.colorbar(cs, ax=[ax],location="bottom", pad=.05,aspect=40,)
        ax.set_aspect('auto', adjustable=None)
        # fig.colorbar(cs, ax=ax, orientation="horizontal", pad=.05)
    if title is not None:
        ax.set_title(title)
    if ax is None:
        # return fig, ax, mm
        return fig, ax, ax
    else:
        # return mm, cs
        return ax, cs


def plotlocmap(
            lat,
            lon,
            ax=None,
            baclat=None,
            baclon=None,
            title=None,
            shape=None,
            txtlabel=None,
            fig=None):
    if ax is None:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.subplots()
    # crsProj = ccrs.PlateCarree()
    # ax = plt.axes(projection=crsProj)
    bounding = [np.min(baclon) - 0.5, np.max(baclon) + 0.5,
                np.min(baclat) - 0.5, np.max(baclat) + 0.5]

    ax.set_extent(bounding, crs=ccrs.Geodetic())
    ax.add_feature(cfeature.OCEAN)  #
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES, linestyle=':')
    # ax.add_feature(cfeature.STATES.with_scale('110m'))

    # mm = basemap.Basemap(
    #     llcrnrlat=min(np.min(baclat),np.min(lat))-0.5,
    #     urcrnrlat=max(np.max(baclat),np.max(lat))+0.5,
    #     llcrnrlon=min(np.min(baclon),np.min(lon))-0.5,
    #     urcrnrlon=max(np.max(baclon),np.max(lon))+0.5,
    #     projection='cyl',
    #     resolution='c',
    #     ax=ax)
    # mm.drawcoastlines()
    # mm.drawstates(linestyle='dashed')
    # mm.drawcountries(linewidth=1.0, linestyle='-.')
    # # x, y = mm(baclon, baclat)
    # # bs = mm.scatter(
    # #     x, y, c='k', s=30)
    # x, y = mm
    x, y = lon, lat
    ax.plot(x, y, 'k*', markersize=12)
    if shape is not None:
        crd = np.array(shape.points)
        par = shape.parts
        if len(par) > 1:
            for k in range(0, len(par) - 1):
                x = crd[par[k]:par[k + 1], 0]
                y = crd[par[k]:par[k + 1], 1]
                # mm.plot(x, y, color='r', linewidth=3)
                ax.plot(x, y, color='r', linewidth=3)
        else:
            y = crd[:, 0]
            x = crd[:, 1]
            # mm.plot(x, y, color='r', linewidth=3)
            ax.plot(x, y, color='r', linewidth=3)

    if title is not None:
        ax.set_title(title, loc='center')
    if txtlabel is not None:
        for ii in range(len(lat)):
            txt = txtlabel[ii]
            xy = (x[ii], y[ii])
            xy = (x[ii]+1.0, y[ii]-1.5)
            ax.annotate(txt, xy, fontsize=18, fontweight='bold')
        if ax is None:
            # return fig, ax, mm
            return fig, ax, ax
        else:
            # return mm
            return ax

def plotPUBloc(data,
            *,
            ax=None,
            lat=None,
            lon=None,
            baclat=None,
            baclon=None,
            title=None,
            cRange=None,
            cRangeint=False,
            shape=None,
            isGrid=False,
            fig = None):

    if cRange is not None:
        vmin = cRange[0]
        vmax = cRange[1]
    else:
        temp = flatData(data)
        vmin = np.percentile(temp, 5)
        vmax = np.percentile(temp, 95)
        if cRangeint is True:
            vmin = int(round(vmin))
            vmax = int(round(vmax))
    if ax is None:
        # fig, ax = plt.figure(figsize=(8, 4))
        fig = plt.figure(figsize=(8, 4))
        ax = fig.subplots()
    # if len(data.squeeze().shape) == 1:
    #     isGrid = False
    # else:
    #     isGrid = True
    bounding = [np.min(baclon) - 0.5, np.max(baclon) + 0.5,
                np.min(baclat) - 0.5, np.max(baclat) + 0.5]

    ax.set_extent(bounding, crs=ccrs.Geodetic())
    ax.add_feature(cfeature.OCEAN)  # ocean background
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES, linestyle=':')
    # ax.add_feature(cfeature.STATES.with_scale('110m'))

    # mm = basemap.Basemap(
    #     llcrnrlat=min(np.min(baclat),np.min(lat))-0.5,
    #     urcrnrlat=max(np.max(baclat),np.max(lat))+0.5,
    #     llcrnrlon=min(np.min(baclon),np.min(lon))-0.5,
    #     urcrnrlon=max(np.max(baclon),np.max(lon))+0.5,
    #     projection='cyl',
    #     resolution='c',
    #     ax=ax)
    # mm.drawcoastlines()
    # mm.drawstates(linestyle='dashed')
    # mm.drawcountries(linewidth=0.5, linestyle='-.')
    # x, y = mm
    x, y = lon, lat

    # bs = mm.scatter(
    #     x, y, c='k', s=30)
    # x, y = mm(lon, lat)

    bs = ax.scatter(x, y, c="k", s=30,)

    if isGrid is True:
        xx, yy = np.meshgrid(x, y)
        # cs = mm.pcolormesh(xx, yy, data, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
        cs = ax.pcolormesh(lon, lat, data, rasterized=True, )
    else:
        # cs = mm.scatter(
        #     x, y, c=data, s=100, cmap=plt.cm.jet, vmin=vmin, vmax=vmax, marker='*')
        cs = ax.scatter(
            x, y, c=data, s=30, cmap=plt.cm.jet, vmin=vmin, vmax=vmax)

    if shape is not None:
        crd = np.array(shape.points)
        par = shape.parts
        if len(par) > 1:
            for k in range(0, len(par) - 1):
                x = crd[par[k]:par[k + 1], 0]
                y = crd[par[k]:par[k + 1], 1]
                # mm.plot(x, y, color='r', linewidth=3)
                ax.plot(x, y, color='r', linewidth=3)
        else:
            y = crd[:, 0]
            x = crd[:, 1]
            # mm.plot(x, y, color='r', linewidth=3)
            ax.plot(x, y, color='r', linewidth=3)
    # mm.colorbar(cs, location='bottom', pad='5%')
    fig.colorbar(cs, ax=[ax], location="bottom", pad=.05, aspect=40, )
    ax.set_aspect('auto', adjustable=None)

    if title is not None:
        ax.set_title(title)
    if ax is None:
        # return fig, ax, mm
        return fig, ax, ax
    else:
        return ax
        # return mm

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
              figsize=[9, 6],
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
        # ax = fig.add_subplot(gs[0:2, k])
        crsProj = ccrs.PlateCarree() # set geographic cooridnates
        ax = fig.add_subplot(gs[0:2, k], projection=crsProj, frameon=True)
        cRange = None if mapColor is None else mapColor[k]
        title = None if mapNameLst is None else mapNameLst[k]
        data = dataMap[k]
        if isGrid is False:
            plotMap(data, lat=lat, lon=lon, fig=fig, ax=ax, cRange=cRange, title=title)
        else:
            grid, uy, ux = utils.grid.array2grid(data, lat=lat, lon=lon)
            plotMap(grid, lat=uy, lon=ux, fig=fig, ax=ax, cRange=cRange, title=title)

    # plot ts
    def onclick(event):
        xClick = event.xdata
        yClick = event.ydata
        d = np.sqrt((xClick - lon)**2 + (yClick - lat)**2)
        ind = np.argmin(d)
        # titleStr = 'pixel %d, lat %.3f, lon %.3f' % (ind, lat[ind], lon[ind])
#         titleStr = 'gage %d, lat %.3f, lon %.3f' % (ind, lat[ind], lon[ind])
#         ax.clear()
#         plotMap(data, lat=lat, lon=lon, ax=ax, cRange=cRange, title=title)
#         ax.plot(lon[ind], lat[ind], 'k*', markersize=12)
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

def plotTsMapGage(dataMap,
              dataTs,
              *,
              lat,
              lon,
              t,
              colorMap=None,
              mapNameLst=None,
              tsNameLst=None,
              figsize=[12, 6]):
    if type(dataMap) is np.ndarray:
        dataMap = [dataMap]
    if type(dataTs) is np.ndarray:
        dataTs = [dataTs]
    nMap = len(dataMap)
    nTs = len(dataTs)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(3, nMap)

    for k in range(nMap):
        ax = fig.add_subplot(gs[0:2, k])
        cRange = None if colorMap is None else colorMap[k]
        title = None if mapNameLst is None else mapNameLst[k]
        data = dataMap[k]
        if len(data.squeeze().shape) == 1:
            plotMap(data, lat=lat, lon=lon, ax=ax, cRange=cRange, title=title)
        else:
            grid, uy, ux = utils.grid.array2grid(data, lat=lat, lon=lon)
            plotMap(grid, lat=uy, lon=ux, ax=ax, cRange=cRange, title=title)
    axTs = fig.add_subplot(gs[2, :])

    def onclick(event):
        xClick = event.xdata
        yClick = event.ydata
        d = np.sqrt((xClick - lon)**2 + (yClick - lat)**2)
        ind = np.argmin(d)
        # titleStr = 'pixel %d, lat %.3f, lon %.3f' % (ind, lat[ind], lon[ind])
        titleStr = 'gage %d, lat %.3f, lon %.3f' % (ind, lat[ind], lon[ind])
        ax.clear()
        plotMap(data, lat=lat, lon=lon, ax=ax, cRange=cRange, title=title)
        ax.plot(lon[ind], lat[ind], 'k*', markersize=12)
        # ax.draw(renderer=None)
        tsLst = list()
        for k in range(nTs):
            tsLst.append(dataTs[k][ind, :])
        axTs.clear()
        plotTS(t, tsLst, ax=axTs, legLst=tsNameLst, title=titleStr)
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
            showDiff='RMSE',
            xlim=None,
            linespec=None):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots()
    else:
        fig = None

    if cLst is None:
        cmap = plt.cm.jet
        cLst = cmap(np.linspace(0, 1, len(xLst)))

    if title is not None:
        ax.set_title(title, loc='left')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    xSortLst = list()
    yRankLst = list()
    rmseLst = list()
    ksdLst = list()
    for k in range(0, len(xLst)):
        x = xLst[k]
        xSort = flatData(x)
        yRank = np.arange(len(xSort)) / float(len(xSort) - 1)
        xSortLst.append(xSort)
        yRankLst.append(yRank)
        if legendLst is None:
            legStr = None
        else:
            legStr = legendLst[k]
        if ref is not None:
            if ref == '121':
                yRef = yRank
            elif ref == 'norm':
                yRef = scipy.stats.norm.cdf(xSort, 0, 1)
            rmse = np.sqrt(((xSort - yRef)**2).mean())
            ksd = np.max(np.abs(xSort - yRef))
            rmseLst.append(rmse)
            ksdLst.append(ksd)
            if showDiff == 'RMSE':
                legStr = legStr + ' RMSE=' + '%.3f' % rmse
            elif showDiff == 'KS':
                legStr = legStr + ' KS=' + '%.3f' % ksd
        ax.plot(xSort, yRank, color=cLst[k], label=legStr, linestyle=linespec[k])
        ax.grid(b=True)
    if xlim is not None:
        ax.set(xlim=xlim)
    if ref == '121':
        ax.plot([0, 1], [0, 1], 'k', label='y=x')
    if ref == 'norm':
        xNorm = np.linspace(-5, 5, 1000)
        normCdf = scipy.stats.norm.cdf(xNorm, 0, 1)
        ax.plot(xNorm, normCdf, 'k', label='Gaussian')
    if legendLst is not None:
        ax.legend(loc='best', frameon=False)
    # out = {'xSortLst': xSortLst, 'rmseLst': rmseLst, 'ksdLst': ksdLst}
    return fig, ax

def plotFDC(xLst,
            *,
            ax=None,
            title=None,
            legendLst=None,
            figsize=(8, 6),
            ref='121',
            cLst=None,
            xlabel=None,
            ylabel=None,
            showDiff='RMSE',
            xlim=None,
            linespec=None):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots()
    else:
        fig = None

    if cLst is None:
        cmap = plt.cm.jet
        cLst = cmap(np.linspace(0, 1, len(xLst)))

    if title is not None:
        ax.set_title(title, loc='center')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    xSortLst = list()
    rmseLst = list()
    ksdLst = list()
    for k in range(0, len(xLst)):
        x = xLst[k]
        xSort = flatData(x, sortOpt=1)
        yRank = np.arange(1, len(xSort)+1) / float(len(xSort) + 1)*100
        xSortLst.append(xSort)
        if legendLst is None:
            legStr = None
        else:
            legStr = legendLst[k]
        if ref is not None:
            if ref == '121':
                yRef = yRank
            elif ref == 'norm':
                yRef = scipy.stats.norm.cdf(xSort, 0, 1)
            rmse = np.sqrt(((xSort - yRef)**2).mean())
            ksd = np.max(np.abs(xSort - yRef))
            rmseLst.append(rmse)
            ksdLst.append(ksd)
            if showDiff == 'RMSE':
                legStr = legStr + ' RMSE=' + '%.3f' % rmse
            elif showDiff == 'KS':
                legStr = legStr + ' KS=' + '%.3f' % ksd
        ax.plot(yRank, xSort, color=cLst[k], label=legStr, linestyle=linespec[k])
        ax.grid(b=True)
    if xlim is not None:
        ax.set(xlim=xlim)
    if ref == '121':
        ax.plot([0, 1], [0, 1], 'k', label='y=x')
    if ref == 'norm':
        xNorm = np.linspace(-5, 5, 1000)
        normCdf = scipy.stats.norm.cdf(xNorm, 0, 1)
        ax.plot(xNorm, normCdf, 'k', label='Gaussian')
    if legendLst is not None:
        ax.legend(loc='best', frameon=False)
    # out = {'xSortLst': xSortLst, 'rmseLst': rmseLst, 'ksdLst': ksdLst}
    return fig, ax


def flatData(x, sortOpt=0):
    # sortOpt: 0: small to large, 1: large to small, -1: no sort
    xArrayTemp = x.flatten()
    xArray = xArrayTemp[~np.isnan(xArrayTemp)]
    if sortOpt == 0:
        xSort = np.sort(xArray)
    elif sortOpt == 1:
        xSort = np.sort(xArray)[::-1]
    elif sortOpt == -1:
        xSort = xArray

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


# import os
import numpy as np
import matplotlib.pyplot as plt
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


def regLinear(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    out = sm.OLS(y, X).fit()
    return out

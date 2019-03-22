import numpy as np


def crd2grid(y, x):
    ux, indX0, indX = np.unique(x, return_index=True, return_inverse=True)
    uy, indY0, indY = np.unique(y, return_index=True, return_inverse=True)

    minDx = np.min(ux[1:] - ux[0:-1])
    minDy = np.min(uy[1:] - uy[0:-1])
    maxDx = np.max(ux[1:] - ux[0:-1])
    maxDy = np.max(uy[1:] - uy[0:-1])
    if maxDx > minDx * 2:
        print("skipped rows")
    #     indMissX=np.where((ux[1:]-ux[0:-1])>minDx*2)[0]
    #     insertX=(ux[indMissX+1]+ux[indMissX])/2
    #     ux=np.insert(ux,indMissX,insertX)
    if maxDy > minDy * 2:
        print("skipped coloums")
    #     indMissY=np.where((uy[1:]-uy[0:-1])>minDy*2)
    #     raise Exception('skipped coloums or rows')

    uy = uy[::-1]
    ny = len(uy)
    indY = ny - 1 - indY
    return (uy, ux, indY, indX)

"""A file to keep all cnn functions"""


def calConvSize(lin, kernel, stride, padding=0, dilation=1):
    lout = (lin + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return int(lout)


def calPoolSize(lin, kernel, stride=None, padding=0, dilation=1):
    if stride is None:
        stride = kernel
    lout = (lin + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return int(lout)


def calFinalsize1d(nobs, noutk, ksize, stride, pool):
    nlayer = len(ksize)
    Lout = nobs
    for ii in range(nlayer):
        Lout = calConvSize(lin=Lout, kernel=ksize[ii], stride=stride[ii])
        if pool is not None:
            Lout = calPoolSize(lin=Lout, kernel=pool[ii])
    Ncnnout = int(Lout * noutk)  # total CNN feature number after convolution
    return Ncnnout

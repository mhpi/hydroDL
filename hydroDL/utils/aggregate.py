# day to year; day to month; month to year
import numpy as np


def day2year(sy, ey, data, nancont):
    """
    :param sy: start year
    :param ey: end year
    :param data: input data, row:day, colum: variable
    :param nancont: the threshold to which control the nan number
                    factor or absolute number
    :return: yearly data: nyear*nvar
    """
    nday, nvar = data.shape
    nyear = ey - sy + 1
    testnday = daynum(sy, ey)
    if nday != testnday:
        raise Exception("The length of input data is not correct")
    sindex = 0
    countyear = 0
    yeardata = np.full((nyear, nvar), np.nan)
    for ii in range(sy, ey + 1):
        temp = yearday(ii)
        if nancont > 1:
            thresnum = nancont
        else:
            thresnum = temp * nancont
        eindex = sindex + temp
        tempdata = data[sindex:eindex, :]
        tempmean = np.nanmean(tempdata, axis=0)
        # deal with the nan data
        # if nan number larger than threshold, give yearly data nan
        nansta = np.sum(np.isnan(tempdata), axis=0)
        tempmean[nansta > thresnum] = np.nan
        yeardata[countyear, :] = tempmean
        countyear = countyear + 1
        sindex = eindex
    if eindex != nday:
        raise Exception("Error happened for the aggregation")
    return yeardata


def day2yearQ(sy, ey, data, nancont, Quantile):
    """
    :param sy: start year
    :param ey: end year
    :param data: input data, row:day, colum: gage
    :param nancont: the threshold to which control the nan number
                    factor or absolute number
    :param Quantile: which quantile to select
    :return: yearly data at Quantile: nyear*nvar
    """
    nday, nvar = data.shape
    nQ = len(Quantile)
    nyear = ey - sy + 1
    testnday = daynum(sy, ey)
    if nday != testnday:
        raise Exception("The length of input data is not correct")
    sindex = 0
    countyear = 0
    yeardata = np.full((nyear, nQ, nvar), np.nan)
    for ii in range(sy, ey + 1):
        temp = yearday(ii)
        if nancont > 1:
            thresnum = nancont
        else:
            thresnum = temp * nancont
        eindex = sindex + temp
        tempdata = data[sindex:eindex, :]
        # deal with the nan data
        # if nan number larger than threshold, give yearly data nan
        nansta = np.sum(np.isnan(tempdata), axis=0)
        tempQ100 = getQ(tempdata)
        qind = [x - 1 for x in Quantile]
        tempQ = tempQ100[qind, :]
        tempQ[:, nansta > thresnum] = np.nan
        yeardata[countyear, :, :] = tempQ
        countyear = countyear + 1
        sindex = eindex
    if eindex != nday:
        raise Exception("Error happened for the aggregation")
    return yeardata


def day2year3d(sy, ey, data, nancont):
    # data: ngage*ntime*nvariable
    # yeardata: nyear*ngage*nvariable
    ngage, nday, nvar = data.shape
    nyear = ey - sy + 1
    yeardata = np.full((nyear, ngage, nvar), np.nan)
    for ii in range(nvar):
        temp = np.swapaxes(data[:, :, ii], 0, 1)
        tempyear = day2year(sy, ey, temp, nancont=nancont)
        yeardata[:, :, ii] = tempyear
    return yeardata


def day2yearQ3d(sy, ey, data, nancont, Quantile):
    # data: ngage*ntime*nvariable
    # yeardata: nyear*ngage*nvariable
    ngage, nday, nvar = data.shape
    nyear = ey - sy + 1
    yeardata = np.full((nyear, ngage, nvar), np.nan)
    for ii in range(nvar):
        temp = np.swapaxes(data[:, :, ii], 0, 1)
        tempyear = day2yearQ(sy, ey, temp, nancont=nancont, Quantile=Quantile)
        yeardata[:, :, ii] = tempyear[:, 0, :]
    return yeardata


def daynum(sy, ey):
    # get the total day number from start to end year
    Nday = 0
    for ii in range(sy, ey + 1):
        temp = yearday(ii)
        Nday = Nday + temp
    return Nday


def yearday(testyear):
    # get the day number of input year
    if (testyear % 4) == 0:
        if (testyear % 100) == 0 and (testyear % 400) != 0:
            temp = 365
        else:
            temp = 366
    else:
        temp = 365
    return temp


def getQ(data):
    # get the 100 quantile flow
    # data = Nday*Ngage
    # return dataQ = 100*Ngage
    Nday, Ngrid = data.shape
    dataQ = np.full([100, Ngrid], np.nan)
    for ii in range(Ngrid):
        tempdata0 = data[:, ii]
        tempdata = tempdata0[~np.isnan(tempdata0)]
        # deal with no data case for some gages
        if len(tempdata) == 0:
            Qflow = np.full(100, np.nan)
        else:
            # sort from small to large
            temp_sort = np.sort(tempdata)
            # select 100 quantile points
            Nlen = len(tempdata)
            ind = np.ceil((np.arange(1, 101) / 100 * Nlen)).astype(int)
            Qflow = temp_sort[ind - 1]
        if len(Qflow) != 100:
            raise Exception("unknown assimilation variable")
        else:
            dataQ[:, ii] = Qflow

    return dataQ

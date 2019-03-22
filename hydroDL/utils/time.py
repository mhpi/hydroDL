import datetime as dt
import numpy as np


def t2dt(t, hr=False):
    tOut = None
    if type(t) is int:
        if t < 30000000 and t > 10000000:
            t = dt.datetime.strptime(str(t), "%Y%m%d").date()
            tOut = t if hr is False else t.datetime()

    if type(t) is dt.date:
        tOut = t if hr is False else t.datetime()

    if type(t) is dt.datetime:
        tOut = t.date() if hr is False else t

    if tOut is None:
        raise Exception('hydroDL.utils.t2dt failed')
    return tOut


def t2dtLst(sd, ed):
    sd = t2dt(sd)
    ed = t2dt(ed)
    tArray = np.arange(sd, ed, dt.timedelta(days=1)).astype(dt.date)
    return tArray


def intersect(tLst1, tLst2):
    C, ind1, ind2 = np.intersect1d(tLst1, tLst2, return_indices=True)
    return ind1, ind2

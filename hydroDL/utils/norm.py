import os
import pandas as pd
import numpy as np
import shutil


def trans_norm(
    data,
    csv_folder_s=None,
    var_s=None,
    from_raw=True,
    is_constant=False,
    csv_path_s=None,
    stat=None,
):
    """
    e.g.
    data = np.arange(12).reshape(1,3,4)
    stat = [90 percentile,10 percentile, mean,std]

    trans_norm(data, stat=stat)
    trans_norm(data, csv_path_s="/data/Statistics/temperature_stat.csv")
    trans_norm(data, "/data", "temperature")
    trans_norm(data, "/data", "soil_texture", is_constant=True)

    # restore
    trans_norm(data, stat=stat, from_raw=False)
    """

    if not csv_folder_s is None:
        if not is_constant:
            csv_path_s = os.path.join(csv_folder_s, "Statistics", var_s + "_stat.csv")
        else:
            csv_path_s = os.path.join(
                csv_folder_s, "Statistics", "const_" + var_s + "_stat.csv"
            )

    if stat is None:
        # stat = [90 percentile,10 percentile, mean,std]

        stat = pd.read_csv(csv_path_s, dtype=float, header=None).values.flatten()

    if from_raw:
        data_out = (data - stat[2]) / stat[3]
    else:
        data_out = data * stat[3] + stat[2]
    return data_out


def re_folder_rec(path_s):
    b = os.path.normpath(path_s).split(os.sep)  # e.g. ['da','jxl','wp']
    b = [x + "/" for x in b]  # e.g. ['da/','jxl/','wp/']
    fn_rec = [
        "".join(b[0 : x[0]]) for x in list(enumerate(b, 1))
    ]  # e.g. ['da/','da/jxl/','da/jxl/wp/']
    fn_None = [os.mkdir(x) for x in fn_rec if not os.path.exists(x)]


def re_folder(path_s, del_old_path=False):
    """
    delete old folder and recreate new one
    e.g. refolder("./tmp/new_path/", del_old_path=True)
    """
    if os.path.exists(path_s):
        if del_old_path:
            shutil.rmtree(path_s)
            re_folder_rec(path_s)
        else:
            pass
    else:
        re_folder_rec(path_s)

def fix_seed(SEED):
    import os
    import numpy as np
    import random
    import torch
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

def cal_statistics(data, re_extreme=True, percent=10):
    """
    data = np.arange(30).reshape([5,6])
    cal_statistics(data)

    re_extreme:
        If re_extreme is True, calculate the value in the middle of 10% and 90% of the data
    percent:
        decide 10% or 90%
    """

    data = data.flatten().astype(np.float)
    # convert -9999 into np.nan
    data[data <= -999] = np.nan
    data = data[~np.isnan(data)]  # remove nan

    left_p10 = np.nanpercentile(data, percent, interpolation="nearest")
    left_p90 = np.nanpercentile(data, 100 - percent, interpolation="nearest")

    if re_extreme:
        """
        There's a bug hidden here that I can't fix. 
        If lb=0, all the zeros will be included, which is the average of 90% of the data
        """
        data_80p = data[(data >= left_p10) & (data <= left_p90)]
    else:
        data_80p = data

    mean = np.nanmean(data_80p)
    eps = 1e-6
    std = np.nanstd(data_80p) + eps
    stat_list = [left_p10, left_p90, mean, std]  # mean for 80% data. std for 80% data.

    return stat_list
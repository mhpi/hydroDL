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

        stat = pd.read_csv(csv_path_s, dtype=np.float, header=None).values.flatten()

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

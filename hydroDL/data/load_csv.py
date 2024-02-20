import os
import pandas as pd
import numpy as np
from hydroDL.utils.norm import trans_norm


class LoadCSV(object):
    def __init__(self, csv_path_s, select_date_list, csv_date_list):

        self.csv_path_s = csv_path_s
        self.select_date_list = select_date_list
        self.csv_date_list = csv_date_list

    def load_time_series(self, var_list, do_norm=True, remove_nan=True, fill_num=0):

        var_type = "var_t"

        for ndx, var_s in enumerate(var_list):

            data_all = self.load_all_csv(
                var_s, var_type, self.select_date_list, self.csv_date_list
            )  # pxiel, time

            # normalization
            if do_norm:
                data_all = trans_norm(
                    data_all, csv_folder_s=self.csv_path_s, var_s=var_s
                )

            # remove nan with 0
            if remove_nan:
                data_all[np.isnan(data_all)] = fill_num

            data_all = data_all[:, :, None]  # pixel, time, 1

            if ndx == 0:
                data = data_all
            else:
                data = np.concatenate((data, data_all), axis=2)  # pixel, time, features

        return data

    def load_constant(
        self,
        var_list,
        do_norm=True,
        remove_nan=True,
        fill_num=0,
        convert_time_series=True,
    ):

        var_type = "var_c"

        for ndx, var_s in enumerate(var_list):

            data_all = self.load_all_csv(
                var_s, var_type, self.select_date_list, self.csv_date_list
            )

            if do_norm:
                data_all = trans_norm(
                    data_all,
                    csv_folder_s=self.csv_path_s,
                    var_s=var_s,
                    is_constant=True,
                )
            if ndx == 0:
                data = data_all
            else:
                data = np.concatenate((data, data_all), axis=-1)  # pixel, features

        # remove nan
        if remove_nan:
            data[np.isnan(data)] = fill_num  # pixel, features

        if convert_time_series:
            data = self.constant_to_time_series(data, self.select_date_list)
        return data

    def load_all_csv(self, var_s, var_type, select_date_list, csv_date_list):

        select_str_s, select_end_s = select_date_list
        select_date_range = pd.date_range(select_str_s, select_end_s)  # e.g. 2015
        select_str_year = int(select_date_range[0].strftime("%Y"))
        select_end_year = int(select_date_range[-1].strftime("%Y"))

        for ndx_year, year_int in enumerate(
            np.arange(select_str_year, select_end_year + 1)
        ):

            if var_type in ["var_t"]:
                path_xdata = os.path.join(
                    self.csv_path_s, str(year_int), var_s + ".csv"
                )
                df = pd.read_csv(path_xdata, dtype=float, header=None)  # pixel, time

                df_a = df.values  # pixel, time
                if ndx_year == 0:
                    data = df_a
                else:
                    data = np.concatenate((data, df_a), axis=1)  # fsg pixel, times

            elif var_type in ["var_c"]:
                path_cData = os.path.join(self.csv_path_s, "const", var_s + ".csv")
                df = pd.read_csv(path_cData, dtype=float, header=None,)  # pixel,

                df_a = df.values  # pixel,
                data = df_a
                data[data == -9999] = np.nan

                return data

        # select data within select_date_range
        csv_str_date_s = csv_date_list[0]
        csv_end_date_s = csv_date_list[-1]

        # create sub time series
        if str(select_str_year) == csv_str_date_s[:4]:
            sub_str_date_s = csv_str_date_s
        else:
            sub_str_date_s = str(select_str_year) + "-01-01"

        if str(select_end_year) == csv_end_date_s[:4]:
            sub_end_date_s = csv_end_date_s
        else:
            sub_end_date_s = str(select_end_year) + "-12-31"

        sub_date_index = pd.date_range(sub_str_date_s, sub_end_date_s)
        ts = pd.Series(np.arange(len(sub_date_index)), index=sub_date_index)
        sub_data = data[:, ts[select_date_range].values]  # pixel, target time
        sub_data[sub_data == -9999] = np.nan
        return sub_data

    def constant_to_time_series(self, data, date_list):
        # data shape: pixel, features
        date_range = pd.date_range(date_list[0], date_list[-1])
        data = data[:, None, :]  # pix, 1, feas
        data = np.repeat(data, len(date_range), axis=1)
        return data

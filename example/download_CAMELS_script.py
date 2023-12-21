
"""
Run this code to download CAMELS data locally and get training_file/validation_file which are pickle files containing numpy floats
change rootDatabase = r"E:\Downloads\CAMELS" below to your download directory. This code works on Linux or Windows.
Afterwards, you can simply use the pickle files (upload to colab if running on colab) defined inline below, e.g.,
train_file = 'training_file' # contains train_x, train_y, train_c as tensors
validation_file = 'validation_file' # contains val_x, val_y, val_c
These variables have not been normalized, and are numpy ndarray as follows:
train_x (forcing data, e.g. precipitation, temperature ...): [pixels, time, features]
train_c (constant data, e.g. soil properties, land cover ...): [pixels, features]
train_y (target variable, e.g. soil moisture, streamflow ...): [pixels, time, 1]
val_x, val_c, val_y
The variables to download, training and test time periods, etc., are defined in the last block of this file. Customize as you wish
"""


import os
import sys
import shutil
import subprocess
import stat
import platform
import os
import requests
import zipfile
from tqdm import tqdm
import pickle

rootDatabase = r"E:\Downloads\CAMELS"
os.chdir(rootDatabase)

"""### git repo"""
def is_running_in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

sys.path.append('../')

def on_rm_error(func, path, exc_info):
    if not os.access(path, os.W_OK):
        # Add write permission and retry
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise
if os.path.exists('hydroDLpack'):
    shutil.rmtree('hydroDLpack', onerror=on_rm_error)
subprocess.run(['git', 'clone', 'https://github.com/mhpi/hydroDL.git', 'hydroDLpack'], check=True)
os.chdir('hydroDLpack')
os.system("conda develop .")
# subprocess.run('conda', 'develop', '.')


def downloadCAMELS():
  if platform.system() == 'Windows':
    def download_file(url, destination):
        """Download a file from a specified URL to a given destination with progress output."""
        try:
            with requests.get(url, stream=True) as response:
                response.raise_for_status()  # Check if the request was successful
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 8192  # 8 Kilobytes
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

                with open(destination, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=block_size):
                        progress_bar.update(len(chunk))
                        file.write(chunk)

                progress_bar.close()

                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")

    def unzip_file(source, destination):
        """Unzip a file from a source to a destination."""
        with zipfile.ZipFile(source, 'r') as zip_ref:
            zip_ref.extractall(destination)

    def check_file_size(file_path, min_size_kb):
        """Check if the file exists and is larger than the specified minimum size in KB."""
        if os.path.exists(file_path):
            file_size_kb = os.path.getsize(file_path) / 1024  # Convert size to KB
            return file_size_kb >= min_size_kb
        return False
    # Base directory
    base_dir = os.getcwd()


    # Create necessary directories using os.path.join for cross-platform compatibility
    os.makedirs(os.path.join(base_dir, 'camels_attributes_v2.0'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'camels_attributes_v2.0', 'camels_attributes_v2.0'), exist_ok=True)

    # Download and unzip the main dataset
    main_dataset_url = 'https://gdex.ucar.edu/dataset/camels/file/basin_timeseries_v1p2_metForcing_obsFlow.zip'
    main_dataset_dest = os.path.join(base_dir, 'basin_timeseries_v1p2_metForcing_obsFlow.zip')

    if not check_file_size(main_dataset_dest, 3326783): # true size is 3,326,784 kb
        print("Trying to download to "+main_dataset_dest)
        print("You can also use your browser to directly download the data to this location so you can proceed ")
        download_file(main_dataset_url, main_dataset_dest)
    else:
        print(main_dataset_dest+" already exists and is larger than 3,326,783 KB. Skipping download.")
        #download_file(main_dataset_url, main_dataset_dest)
    # IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # This programmatic download could fail especially when your internet is slow or UCAR service does not like your connection
    # if you see ERROR, something went wrong, you can use your web browser to directly download https://gdex.ucar.edu/dataset/camels/file/basin_timeseries_v1p2_metForcing_obsFlow.zip
    # and drop the downloaded zip file to the location main_dataset_dest, and proceed with the next steps

    unzip_file(main_dataset_dest, os.path.join(base_dir, 'basin_timeseries_v1p2_metForcing_obsFlow'))

    # List of other files to download
    files_to_download = {
        'https://gdex.ucar.edu/dataset/camels/file/camels_attributes_v2.0.xlsx': 'camels_attributes_v2.0/camels_attributes_v2.0/camels_attributes_v2.0.xlsx',
        'https://gdex.ucar.edu/dataset/camels/file/camels_clim.txt': 'camels_attributes_v2.0/camels_attributes_v2.0/camels_clim.txt',
        'https://gdex.ucar.edu/dataset/camels/file/camels_geol.txt': 'camels_attributes_v2.0/camels_attributes_v2.0/camels_geol.txt',
        'https://gdex.ucar.edu/dataset/camels/file/camels_hydro.txt': 'camels_attributes_v2.0/camels_attributes_v2.0/camels_hydro.txt',
        'https://gdex.ucar.edu/dataset/camels/file/camels_name.txt': 'camels_attributes_v2.0/camels_attributes_v2.0/camels_name.txt',
        'https://gdex.ucar.edu/dataset/camels/file/camels_soil.txt': 'camels_attributes_v2.0/camels_attributes_v2.0/camels_soil.txt',
        'https://gdex.ucar.edu/dataset/camels/file/camels_topo.txt': 'camels_attributes_v2.0/camels_attributes_v2.0/camels_topo.txt',
        'https://gdex.ucar.edu/dataset/camels/file/camels_vege.txt': 'camels_attributes_v2.0/camels_attributes_v2.0/camels_vege.txt'
    }

    # Download additional files
    for url, dest in files_to_download.items():
        full_dest = os.path.join(base_dir, dest)
        download_file(url, full_dest)

    print("Download and unzipping complete.")


def extractCAMELS(Ttrain,attrLst,varF,camels,forType='daymet',flow_regime=1,subset_train="All",file_path=None):
  # flow_regime==1: high flow expert procedures
  train_loader = camels.DataframeCamels(subset=subset_train, tRange=Ttrain, forType=forType)
  x = train_loader.getDataTs(varLst=varF, doNorm=False, rmNan=False, flow_regime=flow_regime)
  y = train_loader.getDataObs(doNorm=False, rmNan=False, basinnorm=False, flow_regime=flow_regime)
  c = train_loader.getDataConst(varLst=attrLst, doNorm=False, rmNan=False, flow_regime=flow_regime)
  # define dataset
  if file_path is not None:
    with open(file_path, 'wb') as f:
      pickle.dump((x, y, c), f)
  return x, y, c

# caution: long-time needed
# you shouldn't need to run this if you are loading directly from pickle file
downloadCAMELS()

import os
import hydroDL
from hydroDL.master import default
from hydroDL.data import camels
import numpy as np

forType = 'daymet'
flow_regime = 1
Ttrain = [19801001, 19951001] #training period
valid_date = [19951001, 20101001]  # Testing period
#define inputs
if forType == 'daymet':
  varF = ['dayl', 'prcp', 'srad', 'tmean', 'vp']
else:
  varF = ['dayl', 'prcp', 'srad', 'tmax', 'vp']

train_file = 'training_file'
validation_file = 'validation_file'
# Define attributes list
attrLst = [ 'p_mean','pet_mean', 'p_seasonality', 'frac_snow', 'aridity', 'high_prec_freq', 'high_prec_dur',
            'low_prec_freq', 'low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
            'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
            'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
            'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class', 'glim_1st_class_frac',
            'geol_2nd_class', 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability']

# a CAMELS dataset-specific object to manage basin normalization
# If you are not using CAMELS, you can discard this one
camels.initcamels(flow_regime=flow_regime, forType=forType, rootDB=rootDatabase)
train_x, train_y, train_c = extractCAMELS(Ttrain,attrLst,varF,camels,forType='daymet',flow_regime=1,subset_train="All",file_path=train_file)
val_x, val_y, val_c = extractCAMELS(valid_date,attrLst,varF,camels,forType='daymet',flow_regime=1,subset_train="All",file_path=validation_file)


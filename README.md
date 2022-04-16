This code contains deep learning code used to model hydrologic systems from soil moisture to streamflow or from projection to forecast. 

[![PyPI](https://img.shields.io/badge/pypi-version%200.1-blue)](https://pypi.org/project/hydroDL/0.1.0/)  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3993880.svg)](https://doi.org/10.5281/zenodo.3993880) [![CodeStyle](https://img.shields.io/badge/code%20style-Black-black)]()


# Installation
There are two different methods for hydroDL installation:

### 1) Using Conda
Create a new environment, then activate it
  ```Shell
conda create -n mhpihydrodl python=3.7
conda activate mhpihydrodl
```

### 2) Using PyPI (stable package)
Install our hydroDL stable package from pip (Python version>=3.7.0)
```
pip install hydroDL
```

### 3) Source latest version
Install our latest hydroDL package from github
```
pip install git+https://github.com/mhpi/hydroDL
```

_Note:_
There exists a small compatibility issue with our code when using the latest pyTorch version. Feel free to contact us if you find any issues or code bugs that you cannot resolve.

# Quick Start:
_The detailed code for quick start can be found in [tutorial_quick_start.py](./example/tutorial_quick_start.py)_

See below for a brief explanation of the major components you need to run a hydroDL model:
```Shell
# imports
from hydroDL.model.crit import RmseLoss
from hydroDL.model.rnn import CudnnLstmModel as LSTM
from hydroDL.model.train import trainModel
from hydroDL.model.test import testModel

# load your training and testing data 
# x: forcing data (pixels, time, features)
# c: attribute data (pixels, features)
# y: observed values (pixels, time, 1)
x_train, c_train, y_train, x_val, c_val, y_val = load_data(...)

# define your model and loss function
model = LSTM(nx=num_variables, ny=1)
loss_fn = RmseLoss()

# train your model
model = trainModel(model,
    x_train,
    y_train,
    c_train,
    loss_fn,
)

# validate your model
pred = testModel(model,
             x_val,
             c_val,
)

```

# Examples

Several examples related to the above papers are presented here. **Click the title link** to see each example.
## [1.Train a LSTM data integration model to make streamflow forecast](example/StreamflowExample-DI.py)
The dataset used is NCAR CAMELS dataset. Download CAMELS following [this link](https://ral.ucar.edu/solutions/products/camels). 
Please download both forcing, observation data `CAMELS time series meteorology, observed flow, meta data (.zip)` and basin attributes `CAMELS Attributes (.zip)`. 
Put two unzipped folders under the same directory, like `your/path/to/Camels/basin_timeseries_v1p2_metForcing_obsFlow`, and `your/path/to/Camels/camels_attributes_v2.0`. Set the directory path `your/path/to/Camels`
as the variable `rootDatabase` inside the code later.

Computational benchmark: training of CAMELS data (w/ or w/o data integration) with 671 basins, 10 years, 300 epochs, in ~1 hour with GPU.

Related papers:  
Feng et al. (2020). [Enhancing streamflow forecast and extracting insights using long‐short term memory networks with data integration at continental scales](https://doi.org/10.1029/2019WR026793). Water Resources Research.

## [2.Train LSTM and CNN-LSTM models for prediction in ungauged regions](example/PUR/trainPUR-Reg.py)
The dataset used is also NCAR CAMELS. Follow the instructions in the first example above to download and unzip the dataset. Use [this code](example/PUR/testPUR-Reg.py) to test your saved models after training finished.

Related papers:  
Feng et al. (2021). [Mitigating prediction error of deep learning streamflow models in large data-sparse regions with ensemble modeling and soft data](https://doi.org/10.1029/2021GL092999). Geophysical Research Letters.  
Feng et al. (2020). [Enhancing streamflow forecast and extracting insights using long‐short term memory networks with data integration at continental scales](https://doi.org/10.1029/2019WR026793). Water Resources Research.

## [3.Train a LSTM model to learn SMAP soil moisture](example/demo-LSTM-Tutorial.ipynb)
The example dataset is embedded in this repo and can be found [here](example/data).
You can also use [this script](example/train-lstm.py) to train model if you don't want to work with Jupyter Notebook.

Related papers:  
Fang et al. (2017), [Prolongation of SMAP to Spatio-temporally Seamless Coverage of Continental US Using a Deep Learning Neural Network](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017GL075619), Geophysical Research Letters.

## [4.Estimate uncertainty of a LSTM network ](example/train-lstm-mca.py)
Related papers:  
Fang et al. (2020). [Evaluating the potential and challenges of an uncertainty quantification method for long short-term memory models for soil moisture predictions](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020WR028095), Water Resources Research.

## [5.Training a multi-scale model](example/multiscale.py)
How to use: [click here](example/multiscale/README.md)

Related papers:  
Liu et al. (2022). [A multiscale deep learning model for soil moisture integrating satellite and in-situ data](https://doi.org/10.1029/2021GL096847), Geophysical Research Letters.

# Citations

If you find our code to be useful, please cite the following papers:

Feng, DP., Lawson, K., and CP. Shen, Mitigating prediction error of deep learning streamflow models in large data-sparse regions with ensemble modeling and soft data, Geophysical Research Letters (2021), https://doi.org/10.1029/2021GL092999

Feng, DP, K. Fang and CP. Shen, Enhancing streamflow forecast and extracting insights using continental-scale long-short term memory networks with data integration, Water Resources Research (2020), https://doi.org/10.1029/2019WR026793

Shen, CP., A trans-disciplinary review of deep learning research and its relevance for water resources scientists, Water Resources Research. 54(11), 8558-8593, doi: 10.1029/2018WR022643 (2018) https://doi.org/10.1029/2018WR022643

Liu, J., Rahmani, F., Lawson, K., & Shen, C. A multiscale deep learning model for soil moisture integrating satellite and in-situ data. Geophysical Research Letters, e2021GL096847 (2022). https://doi.org/10.1029/2021GL096847


Major code contributor: Dapeng Feng (PhD Student, Penn State), Jiangtao Liu(PhD Student., Penn State), Tadd Bindas (PhD Student., Penn State), and Kuai Fang (PhD., Penn State).

# License
hydroDL has a Non-Commercial license, as found in the [LICENSE](./LICENSE) file.



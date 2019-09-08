# Citations

If you find our code to be useful, please cite the following papers:

K. Fang, M. Pan, and CP. Shen, [The value of SMAP for long-term soil moisture estimation with the help of deep learning], Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2018.2872131 (2018) https://ieeexplore.ieee.org/document/8497052

K. Fang, CP. Shen, D. Kifer and X. Yang, [Prolongation of SMAP to Spatio-temporally Seamless Coverage of Continental US Using a Deep Learning Neural Network], Geophysical Research Letters, doi: 10.1002/2017GL075619, preprint accessible at: arXiv:1707.06611 (2017) https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2017GL075619

Shen, CP., [A trans-disciplinary review of deep learning research and its relevance for water resources scientists], Water Resources Research. 54(11), 8558-8593, doi: 10.1029/2018WR022643 (2018) https://doi.org/10.1029/2018WR022643


# Example
Two examples with sample data are wrapped up including
 - [train a LSTM network to learn SMAP soil moisture](example/train-lstm.py)
 - [estimate uncertainty of a LSTM network ](example/train-lstm-mca.py)

A demo for temporal test is [here](example/demo-temporal-test.ipynb)


# Database description
## Database Structure
```
├── CONUS
│   ├── 2000
│   │   ├── [Variable-Name].csv
│   │   ├── ...
│   │   ├── timeStr.csv
│   │   └── time.csv
│   ├── ...
│   ├── 2017
│   │   └── ...
│   ├── const
│   │   ├── [Constant-Variable-Name].csv
│   │   └── ...
│   └── crd.csv
├── CONUSv4f1
│   └── ...
├── Statistics
│   ├── [Variable-Name]_stat.csv
│   ├── ...
│   ├── const_[Constant-Variable-Name]_stat.csv
│   └── ...
├── Subset
│   ├── CONUS.csv
│   └── CONUSv4f1.csv
└── Variable
    ├── varConstLst.csv
    └── varLst.csv
```
### 1. Dataset folders (*CONUS* , *CONUSv4f1*)
Data folder contains all data including both training and testing, time-dependent variables and constant variables. 
In example data structure, there are two dataset folders - *CONUS* and *CONUSv4f1*. Those data are saved in:

 - **year/[Variable-Name].csv**:

A csv file of size [#grid, #time], where each column is one grid and each row is one time step. This file saved data of a time-dependent variable of current year. For example, *CONUS/2010/SMAP_AM.csv* is SMAP data of 2002 on the CONUS. 

Most time-dependent varibles comes from NLDAS, which included two forcing product (FORA, FORB) and three simulations product land surface models (NOAH, MOS, VIC). Variables are named as *[variable]\_[product]\_[layer]*, and reference of variable can be found in [NLDAS document](https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/README.NLDAS2.pdf). For example, *SOILM_NOAH_0-10* refers to soil moisture product simulated by NOAH model at 0-10 cm. 

Other than NLDAS, SMAP data are also saved in same format but always used as target. In level 3 database, there are two SMAP csv files which are only available after 2015: *SMAP_AM.csv* and *SMAP_PM.csv*. 

-9999 refers to NaN. 

- **year/time.csv** & **timeStr.csv**

Dates csv file of current year folder, of size [#date]. *time.csv* recorded Matlab datenum and *timeStr.csv* recorded date in format of yyyy-mm-dd.

Notice that each year start from and end before April 1st. For example data in folder 2010 is actually data from 2010-04-01 to 2011-03-31. The reason is that SMAP launched at April 1st. 

- **const/[Constant Variable Name].csv**

csv file for constant variables of size [#grid]. 

- **crd.csv**

Coordinate of all grids. First Column is latitude and second column is longitude. Each row refers a grid.

### 2. Statistics folder

Stored statistics of variables in order to do data normalization during training. Named as:
- Time dependent variables-> [variable name].csv
- Constant variables-> const_[variable name].csv

Each file wrote four statistics of variable:
- 90 percentile
- 10 percentile
- mean
- std

During training we normalize data by (data - mean) / std

### 3. Subset folder
Subset refers to a subset of grids from the complete dataset (CONUS or Global). For example, a subset only contains grids in Pennsylvania. All subsets (including the CONUS or Global dataset) will have a *[subset name].csv* file in the *Subset* folder. *[subset name].csv* is wrote as:
- line 1 -> root dataset 
- line 2 - end -> indexs of subset grids in rootset (start from 1)

If the index is -1 means all grid, from example CONUS dataset. 

### 4. Variable folder
Stored csv files contains a list of variables. Used as input to training code. Time-dependent variables and constant variables should be stored seperately. For example:
- varLst.csv -> a list of time-dependent variables used as training predictors.
- varLst.csv -> a list of constant variables used as training predictors.

This code contains deep learning code used to modeling hydrologic systems, from soil moisture to streamflow, from projection to forecast. 

This released code depends on our hydroDL repository, please follow our original github repository where we will release new updates occasionally
https://github.com/mhpi/hydroDL
# Citations

If you find our code to be useful, please cite the following papers:

Feng, DP., Lawson, K., and CP. Shen, Mitigating prediction error of deep learning streamflow models in large data-sparse regions with ensemble modeling and soft data, Geophysical Research Letters (2021), https://doi.org/10.1029/2021GL092999

Feng, DP, K. Fang and CP. Shen, Enhancing streamflow forecast and extracting insights using continental-scale long-short term memory networks with data integration, Water Resources Research (2020), https://doi.org/10.1029/2019WR026793

Shen, CP., A trans-disciplinary review of deep learning research and its relevance for water resources scientists, Water Resources Research. 54(11), 8558-8593, doi: 10.1029/2018WR022643 (2018) https://doi.org/10.1029/2018WR022643

Major code contributor: Dapeng Feng (PhD Student, Penn State) and Kuai Fang (PhD., Penn State)

# Examples
The environment we are using is shown as the file `repoenv.yml`. To create the same conda environment, please run:
  ```Shell
conda env create -f repoenv.yml
```
Activate the installed environment before running the code:
  ```Shell
conda activate mhpihydrodl
```
You can also use this `Environment Setup_Tutorial.pdf` document as a reference to set up your environment and solve some frequently encountered questions. 
There may be a small compatibility issue with our code when using very high pyTorch version. Welcome to contact us if you find any issue not able to solve or bug.


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
# License
Non-Commercial Software License Agreement

By downloading the hydroDL software (the “Software”) you agree to
the following terms of use:
Copyright (c) 2020, The Pennsylvania State University (“PSU”). All rights reserved.

1. PSU hereby grants to you a perpetual, nonexclusive and worldwide right, privilege and
license to use, reproduce, modify, display, and create derivative works of Software for all
non-commercial purposes only. You may not use Software for commercial purposes without
prior written consent from PSU. Queries regarding commercial licensing should be directed
to The Office of Technology Management at 814.865.6277 or otminfo@psu.edu.
2. Neither the name of the copyright holder nor the names of its contributors may be used
to endorse or promote products derived from this software without specific prior written
permission.
3. This software is provided for non-commercial use only.
4. Redistribution and use in source and binary forms, with or without modification, are
permitted provided that redistributions must reproduce the above copyright notice, license,
list of conditions and the following disclaimer in the documentation and/or other materials
provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS &quot;AS IS&quot;
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

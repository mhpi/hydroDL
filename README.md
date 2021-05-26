This code contains deep learning code used to modeling hydrologic systems, from soil moisture to streamflow, from projection to forecast. 

This released code depends on our hydroDL repository, please follow our original github repository where we will release new updates occasionally
https://github.com/mhpi/hydroDL
# Citations

If you find our code to be useful, please cite the following papers:

Feng, DP., Lawson, K., and CP. Shen, Mitigating prediction error of deep learning streamflow models in large data-sparse regions with ensemble modeling and soft data, Geophysical Research Letters (2021, Accepted) arXiv preprint https://arxiv.org/abs/2011.13380

Feng, DP, K. Fang and CP. Shen, Enhancing streamflow forecast and extracting insights using continental-scale long-short term memory networks with data integration], Water Resources Research (2020), https://doi.org/10.1029/2019WR026793

Shen, CP., A trans-disciplinary review of deep learning research and its relevance for water resources scientists, Water Resources Research. 54(11), 8558-8593, doi: 10.1029/2018WR022643 (2018) https://doi.org/10.1029/2018WR022643

Major code contributor: Dapeng Feng (PhD Student, Penn State) and Kuai Fang (PhD., Penn State)

# Examples
The environment to repeat our results is shown as the file `repoenv.yml`. To create the same conda environment we are using, please run:
  ```Shell
conda env create -f repoenv.yml
```
Activate the installed environment before running the code:
  ```Shell
conda activate mhpihydrodl
```
You can also use this `Environment Setup_Tutorial.pdf` document as a reference to set up your environment and solve some frequently encountered questions. 
Note that there may be a compatibility issue with our code when using pyTorch version higher than 1.2. If you have to use high pyTorch version and indeed encounter an AttributeError not able to solve, please contact us to get instructions for this issue. Only a small modification is needed.


Several examples related to the above papers are presented here. Click the title link to see each example.
## [Train a LSTM model to learn SMAP soil moisture](example/demo-LSTM-Tutorial.ipynb)
The example dataset is embedded in this repo and can be found [here](example/data)
Can also use [this script](example/train-lstm.py) to train model if you don't want to work with Jupyter Notebook

## [Train a LSTM data integration model to make streamflow forecast](example/StreamflowExample-Integ.py)
The dataset used is NCAR CAMELS dataset. Download CAMELS following [this link](https://ral.ucar.edu/solutions/products/camels). Video explanations of this example and how to read the dataset is [here](https://psu.zoom.us/rec/play/uJUtduv9pzk3SdyQ4wSDC_J_W9ToLv6sgCFP-aZcnRq2USEFMVSuYOBHMLRXdUH9qR2nWoRsIOxzLc0G?startTime=1579703281000). We describe an batch-enabled interface using master files.

Computational benchmark: training of CAMELS data (w/ or w/o data integration) with 671 basins, 10 years, 300 epochs, in ~1 hour with GPU.

## [Train LSTM and CNN-LSTM models for prediction in ungauged regions](example/PUR/trainPUR-HUC-All.py)
The dataset used is also NCAR CAMELS as above. Use [this code](example/PUR/testPUR-HUC-All.py) to test your saved models after training finished.

## [Estimate uncertainty of a LSTM network ](example/train-lstm-mca.py)

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

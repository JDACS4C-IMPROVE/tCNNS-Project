# tCNNS-Project
Twin convolutional neural network for drugs in SMILES format.

**tCNNS** is a single-drug response prediction model that uses drug and genomic features.  

This model has been curated as a part of the [_IMPROVE Project_](https://github.com/JDACS4C-IMPROVE). 
The original code can be found [here](https://github.com/Lowpassfilter/tCNNS-Project).  
See [Reference](#reference) for the original paper.

## Model

See [Model](READMEs/Model.md) for more details.

## Data

See [Data](READMEs/Data.md) for more details.

## Requirements -> Dependencies

- `conda`
or
- `singularity`

## Source Codes
## Datasets

## Installation

### With Conda

Create a conda environment:
```sh
conda create -n tcnns python=3.6.9 tensorflow-gpu=1.15 -y
```

Activate the environment:
```sh
conda activate tcnns
```

Clone the repo:
```sh
git clone https://github.com/JDACS4C-IMPROVE/tCNNS-Project.git
cd tCNNS-Project
git checkout develop
```

Install CANDLE package:
```sh
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
```

Clone the `IMPROVE` repo to a directory of your preference:
```sh
cd ..
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
cd IMPROVE/
git checkout develop
```

To use the IMPROVE libraries and scripts, set the environment variable like so:
```sh
export PYTHONPATH=$PYTHONPATH:/lambda_stor/homes/ac.sejones/test/IMPROVE
```

Navigate to the model repo:
```sh
cd ../tCNNS-Project
```
### With Singularity

See Singularity readme...

## Model Workflow -> Step-by-step running 

Different source files and target files can be used to produce a CSA of tCNNS models. Specify the datasets to be trained and tested on in the model config file (i.e. `tcnns_csa_params.txt`):

*TO DO: Link to page that describes CSA? Ignore bash scripts?*

```
[Global_Params]
model_name = "CSA_tCNNS"
source_data = ["gCSI"]
target_data = ["gCSI"]
split_ids = [3, 7]
model_config = "tcnns_csa_params.txt"
```

### 1. Download data and define required environment variable

Download the CSA benchmark data into the model directory from https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/:

```bash
wget --cut-dirs=8 -P ./ -nH -np -m https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/
```

Set environment variables:
```bash
export IMPROVE_DATA_DIR="./csa_data/"
export PYTHONPATH=$PYTHONPATH:/lambda_stor/homes/ac.sejones/test/IMPROVE
```

### 2. Proprocess raw data

```bash
python tcnns_preprocess_improve.py --config tcnns_csa_params.txt
```

*Make sure that the model config file is in `IMPROVE_DATA_DIR`.*

This script will produce the following files:
```
out_tCNNS/
├── processed
│   ├── test_data.pt
│   ├── train_data.pt
│   └── val_data.pt
├── test_y_data.csv
├── train_y_data.csv
└── val_y_data.csv
```

### 3. Train model
```bash
python tcnns_train_improve.py --config tcnns_csa_params.txt
```

This trains a tCNNS model using the processed data. By default, this model uses early stopping.

### 4. Infer with trained model

```bash
python tcnns_infer_improve.py --config tcnns_csa_params.txt
```

The scripts uses processed data and the trained model to evaluate performance found in the following files: `val_scores.json` and `val_predicted.csv`.


## Changing hyperparameters

Hyperparameters of the model can be adjusted in the config file `tcnns_default_model.txt`. 

A different config file with the same variables can be called by adding a new environment variable: 

```sh
export CANDLE_CONFIG=tcnns_benchmark_model.txt
bash train.sh $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR $CANDLE_CONFIG
```

To get more info on the hyperparameters, refer to [tcnns.py](tcnns.py) or run the following command:
```
tcnns_train_improve.py --help
```

## Cross-Study Analysis (CSA) Workflow

Describe how to run
major steps

## HPO/Supervisor

How to run

## Reference
Liu, P., Li, H., Li, S., & Leung, K. S. (2019). Improving prediction of phenotypic drug response on cancer cell lines using deep convolutional network. BMC bioinformatics, 20(1), 408. https://doi.org/10.1186/s12859-019-2910-6

See Reproducibility to reproduce results.


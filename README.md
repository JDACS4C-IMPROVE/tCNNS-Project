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

## Requirements

- `conda`
or
- `singularity`

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

Model definition file `tCNNS.def` is located [here](https://github.com/JDACS4C-IMPROVE/Singularity/blob/develop/definitions/tCNNS.def). 

Clone IMPROVE/Singularity repo:
```sh
git clone https://github.com/JDACS4C-IMPROVE/Singularity.git
cd Singularity
```

Build Singularity:
```sh
mkdir images
singularity build --fakeroot images/tCNNS.sif definitions/tCNNS.def 
```

## Example Usage 

### With Conda

Environment variables:

 * `CANDLE_DATA_DIR` - path to data, model, and results directory
 * `CUDA_VISIBLE_DEVICES` - which GPUs should be used

Set environment variables:
```sh
export CANDLE_DATA_DIR=candle_data_dir
export CUDA_VISIBLE_DEVICES=0
```

Preprocess:
```sh
bash preprocess.sh $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR
```

Train:
```sh
bash train.sh $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR
```

Infer:
```sh
bash infer.sh $CUDA_VISIBLE_DEVICES $CANDLE_DATA_DIR
```

### With Singularity

To use the container, you must make your `CANDLE_DATA_DIR` available inside the container as `/candle_data_dir`.

Environment variables:

 * `CANDLE_DATA_DIR` - path to data, model, and results directory
 * `CONTAINER` - path and name of image file
 * `CUDA_VISIBLE_DEVICES` - which GPUs should be used

Singularity options:

 * `--nv` - enable Nvidia support
 * `--bind` - make the directory available inside container

Set environment variables:
```sh
export CANDLE_DATA_DIR=candle_data_dir
export CONTAINER=tCNNS.sif
export CUDA_VISIBLE_DEVICES=0
```

Preprocess:
```sh
singularity exec --nv --bind $CANDLE_DATA_DIR:/candle_data_dir $CONTAINER preprocess.sh $CUDA_VISIBLE_DEVICES /candle_data_dir 
```

Train:
```sh
singularity exec --nv --bind $CANDLE_DATA_DIR:/candle_data_dir $CONTAINER train.sh $CUDA_VISIBLE_DEVICES /candle_data_dir 
```

Infer:
```sh
singularity exec --nv --bind $CANDLE_DATA_DIR:/candle_data_dir $CONTAINER infer.sh $CUDA_VISIBLE_DEVICES /candle_data_dir 
```

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

Alternatively, one can modify the hyperparameters on the command line like so:

```sh
singularity exec --nv --bind $CANDLE_DATA_DIR:/candle_data_dir $CONTAINER train.sh $CUDA_VISIBLE_DEVICES /candle_data_dir --epochs 1
```

## Reproducing original results

The `tcnns_default_model.txt` contains the hyperparameters that were used in the [original paper](#reference). This config file is the default file for the following scripts.   

Both the raw and processed data are available [here](https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/tCNNS/). See [Data](READMEs/Data.md) for more details on the original data.  

```sh
# set environment variable to point to data folder
export CANDLE_DATA_DIR=candle_data_dir

# preprocess raw data
python tcnns_preprocess_improve.py

# train model
python tcnns_train_improve.py

# infer with trained model on test data
python tcnns_infer_improve.py
```

## Cross-Study Analysis (CSA) Workflow

Different source files and target files can be used to produce a CSA of tCNNS models. Specify the datasets to be trained and tested on in the model config file (i.e. `tcnns_csa_params.txt`):

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
wget --cut-dirs=7 -P ./ -nH -np -m ftp://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data
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

## Reference
Liu, P., Li, H., Li, S., & Leung, K. S. (2019). Improving prediction of phenotypic drug response on cancer cell lines using deep convolutional network. BMC bioinformatics, 20(1), 408. https://doi.org/10.1186/s12859-019-2910-6


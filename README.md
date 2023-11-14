# tCNNS-Project
Twin convolutional neural network for drugs in SMILES format.

**tCNNS** is a single-drug response prediction model that uses drug and genomic features.  

This model has been curated as a part of the [_IMPROVE Project_](https://github.com/JDACS4C-IMPROVE).  
The original code can be found [here](https://github.com/Lowpassfilter/tCNNS-Project).  
See [Reference](#reference) for the original paper.

## TODO

Different ways to train model - epochs, early stopping  
Downloading data  
Benchmark vs. original  
Preprocessing of AUC vs IC50  
Help or config definition section

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
```

Install CANDLE package:
```sh
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
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

Alternatively, one can modify the hyperparameters on the command line like so:

```sh
singularity exec --nv --bind $CANDLE_DATA_DIR:/candle_data_dir $CONTAINER train.sh $CUDA_VISIBLE_DEVICES /candle_data_dir --epochs 1
```

## Reference
Liu, P., Li, H., Li, S., & Leung, K. S. (2019). Improving prediction of phenotypic drug response on cancer cell lines using deep convolutional network. BMC bioinformatics, 20(1), 408. https://doi.org/10.1186/s12859-019-2910-6


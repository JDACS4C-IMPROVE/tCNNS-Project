# tCNNS-Project
Twin convolutional neural network for drugs in SMILES format.

This model is curated as a part of the [_IMPROVE Project_](https://github.com/JDACS4C-IMPROVE).

The original code is [_here_](https://github.com/Lowpassfilter/tCNNS-Project).

## Model

See [Model](READMEs/Model.md) for more details.

## Data

See [Data](READMEs/Data.md) for more details.

## Requirements

- `conda`
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

Install CANDLE package:
```sh
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
```

### With Singularity

Model definition file 'tCNNS.def' is located [_here_](https://github.com/JDACS4C-IMPROVE/Singularity/tree/develop/definitions). 

Build Singularity:
```sh
singularity build --fakeroot tCNNS.sif tCNNS.def 
```

## Example Usage 

Set environment variable for folder to hold data, model, and results:
```sh
export CANDLE_DATA_DIR=candle_data_dir
```

### With Conda

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

To use the container, you must make your data directory available inside the container as `/candle_data_dir`.

Environment variables:

 * `CANDLE_DATA_DIR` - path to data directory
 * `CONTAINER` - path and name of image file
 * `CUDA_VISIBLE_DEVICES` - which GPUs should be used

Singularity options:

 * `--nv` - enable Nvidia support
 * `--bind` - make the directory available inside container

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


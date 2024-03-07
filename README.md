# tCNNS-Project
Twin convolutional neural network for drugs in SMILES format.

**tCNNS** is a single-drug response prediction model that uses drug and genomic features.  
This model has been curated as a part of the [_IMPROVE Project_](https://github.com/JDACS4C-IMPROVE).  

The original code can be found [here](https://github.com/Lowpassfilter/tCNNS-Project).  
See [Reference](#reference) for the original paper.
See [Model](READMEs/Model.md) for more details.
See [Data](READMEs/Data.md) for more details.
See [Reproducing](READMEs/Reproducing.md) to reproduce the original results of the paper in the IMPROVE framework.
See [HPO](READMEs/HPO.md) for how to change the hyperparameters.
See [CSA](READMEs/CSA.md) for details on how to run the Cross-Study Analysis workflow.

## Requirements

- `conda`
or
- `singularity`

## Installation
See [Singularity](READMEs/Singularity.md) for installation with Singularity.

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



## Example Usage 
See [Singularity](READMEs/Singularity.md) for usage with Singularity.
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






## Reference
Liu, P., Li, H., Li, S., & Leung, K. S. (2019). Improving prediction of phenotypic drug response on cancer cell lines using deep convolutional network. BMC bioinformatics, 20(1), 408. https://doi.org/10.1186/s12859-019-2910-6


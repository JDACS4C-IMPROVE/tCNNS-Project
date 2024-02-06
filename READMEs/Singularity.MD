# Singularity

Describe how to create and run Singularity


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

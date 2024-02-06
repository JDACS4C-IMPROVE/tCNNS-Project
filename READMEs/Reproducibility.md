# Reproducibility

Describe the steps to reproduce
Results

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

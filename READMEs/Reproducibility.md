# Reproducing original results

The `tcnns_default_model.txt` contains the hyperparameters that were used in the [original paper](#reference). This config file is the default file for the following scripts.   

The original data can be found [here](https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/tCNNS/). See [Data](Data.md) for more details on the original data.  

```diff
+ Can also point to data in original GitHub or MoDaC?
```

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

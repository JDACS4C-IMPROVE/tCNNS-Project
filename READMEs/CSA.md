# Cross-Study Analysis (CSA) Workflow

Different source files and target files can be used to produce a CSA of tCNNS models. Specify the datasets to be trained and tested on in the model config file (i.e. `tcnns_csa_params.txt`):

TO DO: Link to page that describes CSA?

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

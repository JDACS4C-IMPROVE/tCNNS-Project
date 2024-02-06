Deep learning (DL) models built using popular DL frameworks can take various types of data from simple CSV to more complex structures such as `.pt` with PyTorch and `TFRecords` with TensorFlow.
Constructing datasets for drug response prediction (DRP) models generally requires combining heterogeneous data such as cancer and drug information and treatment response values.
We distinguish between two types of data:
- __ML data__. Data that can be directly consumed by prediction models for training and testing (e.g., `TFRecords`).
- __Raw data__. Data that are used to generate ML data (e.g., treatment response values, cancer and drug info). These usually include data files from drug sensitivity studies such as CCLE, CTRP, gCSI, GDSC, etc.

As part of model curation, the original data that is provided with public DRP models is copied to an FTP site. The full path is https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/ . For each model, a subdirectory is created for storing the model's data.

The raw data and ML data are located, respectively, in `data` and `data_processed` folders. E.g., the data for tCNNS can be found in this FTP location: https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/tCNNS/


Preprocessing scripts are often required to generate ML data from raw data. However, not all public repositories provide the necessary scripts.


# Raw data
The raw data is downloaded from GDSC website (version 6.0) and refers here to three types of data:
1) Dose-independent drug response values.
`PANCANCER_IC.csv`: drug and cell IDs, IC50 values and other metadata (223 drugs and 948 cell lines).
2) Cancer sample information. `PANCANCER_Genetic_feature.csv`: 735 binary features that include mutations and copy number alterations.
3) Drug information. `drug_smiles.csv`: SMILES strings of drug molecules. The canonical SMILES were retrieved from PubChem using CIDs (`Druglist.csv`) or from LINCS if not available in PubChem. The script `preprocess.py` provides functions to download CIDs and SMILES using PubChem's APIs.

The raw data is available in this FTP location: https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/tCNNS/tcnns_data.tar.gz


# ML data
The script `preprocess.py` uses raw data to generate ML data that can be used to train and test with tCNNS. The necessary raw data are automatically downloaded from the FTP server using a `candle_lib` utility function `get_file()` and processed:

- __Response data__. IC50 values (`PANCANCER_IC.csv`) are normalized in the (0,1) interval.
- __Cancer features__. 735 binary features, including mutations and copy number alterations, are not modified.
- __Drug features__. SMILES string of each drug is converted into a 28 by 188 one-hot matrix where a value 1 at row i and column j means that the ith symbol appears at jth position in the SMILES string.


The ML data files are available in this FTP location: https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/tCNNS/tcnns_data_processed.tar.gz. 

These files can be automatically downloaded from the FTP server using the `candle_lib` utility function `get_file()`.

# Cross-study analysis data preprocessing 

# Using your own data
Ultimately, we want to be able to train models with other datasets (not only the ones provided with the model repo). This requires the preprocessing scripts to be available and reproducible.

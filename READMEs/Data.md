# Data

## Original data

As part of model curation, the original data that is provided with public DRP models is copied to Argonne National Lab's FTP site. The full path is https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/. For each model, a subdirectory is created for storing the model's data.

The original data was downloaded from GDSC website (version 6.0) and refers here to three types of data:
1) Dose-independent drug response values.
`PANCANCER_IC.csv`: drug and cell IDs, IC50 values and other metadata (223 drugs and 948 cell lines).
2) Cancer sample information. `PANCANCER_Genetic_feature.csv`: 735 binary features that include mutations and copy number alterations.
3) Drug information. `drug_smiles.csv`: SMILES strings of drug molecules. The canonical SMILES were retrieved from PubChem using CIDs (`Druglist.csv`) or from LINCS if not available in PubChem. The authors' original script provides functions to download CIDs and SMILES using PubChem's APIs.

The original data is available at this FTP location: https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/tCNNS/tcnns_data.tar.gz

```diff
+ Should we point to MoDaC? Not all models have their data on the FTP site.
```

## IMPROVE Benchmark Dataset

TO DO: list the files used from the IMPROVE Benchmark Dataset and how to access data (modac, ftp)

## Preprocessing raw data
The script `tcnns_preprocess_improve.py` generates preprocessed data that can be used to train validate, and test with the tCNNS model. The following data are generated:

- __Response data__. AUC values or IC50 values normalized in the (0,1) interval. For the IMPROVE Benchmark Dataset, the IC50 values had to be TO DO.
- __Cancer features__. Over 700 binary features indicating the presense of mutations and copy number alterations based on the features described in the [Genomics of Drug Sensitivity in Cancer website](https://www.cancerrxgene.org/features). Note that not all of the binary features from the original paper could be replicated entirely.
- __Drug features__. SMILES string of each drug is converted into a X by X one-hot matrix where a value 1 at row i and column j means that the ith symbol appears at jth position in the SMILES string.






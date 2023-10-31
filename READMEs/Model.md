# tCNNS
Twin convolutional neural network for drugs in SMILES format.


## Structure
tCNNS consists of two 1D convolutional neural network (CNN) branches for distilling the features for drugs and cell lines separately. For the cell-line branch, the input data are 1D feature vectors of 735 genetic features related to mutation state or copy number alteration. 
For the drug branch, the input data are one-hot matrices (28 unique symbols x 188 positions) for each drug where a value of 1 at row i and column j means that the ith symbol appears at jth position in the SMILES format for that drug. The structures for the two branches are the same. For each branch, there are three similar layers: each layer with convolution width 7, convolution stride 1, max pooling width 3, and pooling stride 3. The only difference between the layers is that their number of channels are 40, 80 and 60, respectively. After the two branches of the CNN, there is a fully connected network (FCN), which aims to do the regression analysis between the output of the two branches and the IC50 values. There are three hidden layers in the FCN, each with 1024 neurons. The dropout probability is set to be 0.5 for the FCN during the training phase.


## Data sources
The primary data sources that have been used to construct datasets for model training and testing (i.e., ML data) include:
- GDSC version 6.0 - cell line and drug IDs, treatment response, cell line omics data
- PubChem - drug SMILES
- Library of the Integrated Network-based Cellular Signatures (LINCS) - drug SMILES


## Data and preprocessing
Cancer cell line (CCL) omics data and treatment response data (IC50) were originally downloaded from the GDSC website. Th canonical SMILES of 223 drugs were obtained from either PubChem or LINCS. Refer to [Data.md](Data.md) for more info regarding the raw data provided with the original tCNNS model repo and the preprocessing scripts to generate ML data for model training and testing.


## Evaluation
Several evaluation schemes were used for the analysis of prediction performance.

- Mixed set: Cell lines and drugs can appear in train, validation, and test sets.
- Cell-blind: No overlap on cell lines in train, validation, and test sets.
- Drug-blind: No overlap on drugs in train, validation, and test sets. 
- Exclusion of extrapolated activity data: The model is trained and tested on a subset of GDSC data referred to as max_conc data. This subset of data only includes IC50 values below the maximum screening concentration (max_conc), which is the maximum IC50 value verified by biological experiments for each drug in GDSC.
- Number of cell line features: The performance of the model is tested with different numbers of cell line features.
- Modification of SMILES format:  The one-hot representation of the SMILES format was modified in three ways for assessing whether the model captures the biological meaning in the data versus the statistical pattern of the data.


## URLs
- [Original GitHub](https://github.com/Lowpassfilter/tCNNS-Project)
- [IMPROVE GitHub](https://github.com/JDACS4C-IMPROVE/tCNNS-Project/tree/develop)
- [Original Data](https://ftp.mcs.anl.gov/pub/candle/public/improve/model_curation_data/tCNNS/)


## Reference
Liu, P., Li, H., Li, S., & Leung, K. S. (2019). Improving prediction of phenotypic drug response on cancer cell lines using deep convolutional network. BMC bioinformatics, 20(1), 408. https://doi.org/10.1186/s12859-019-2910-6

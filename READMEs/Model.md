## Structure
The original structure of tCNNS consists of two 1D convolutional neural network (CNN) branches for distilling the features for drugs and cell lines separately. For the cell-line branch, the input data are 1D feature vectors of 735 genetic features related to mutation state or copy number alteration. 
For the drug branch, the input data are one-hot matrices (28 unique symbols x 188 positions) for each drug where a value of 1 at row i and column j means that the ith symbol appears at jth position in the SMILES format for that drug. The structures for the two branches are the same. For each branch, there are three similar layers: each layer with convolution width 7, convolution stride 1, max pooling width 3, and pooling stride 3. The only difference between the layers is that their number of channels are 40, 80 and 60, respectively. After the two branches of the CNN, there is a fully connected network (FCN) that takes the output of the two branches to predict the IC50 values for each cell line and drug combination. There are three hidden layers in the FCN, each with 1024 neurons. The dropout probability is set to be 0.5 for the FCN during the training phase.

```diff
+ Optional - add image from paper
```

## IMPROVE modifications
TO DO - Describe any modifications to the code to take in other drug response predication metrics (ex. AUC) or train the model. Say no modifications if changes weren't needed. 






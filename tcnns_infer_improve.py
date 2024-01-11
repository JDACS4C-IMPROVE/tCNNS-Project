import os
from pathlib import Path
import candle
import tensorflow as tf
from batcher import *
import tcnns
import numpy as np
import pandas as pd
import math
import time
import sys
from typing import Dict

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
from improve.metrics import compute_metrics

# [Req] Imports from preprocess and train scripts
from tcnns_preprocess_improve import preprocess_params
from tcnns_train_improve import metrics_list, train_params

# get file path of script
filepath = Path(__file__).resolve().parent # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_infer_params
# 2. model_infer_params
#
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific params in this script.
app_infer_params = []

# 2. Model-specific params (Model: LightGBM)
# All params in model_infer_params are optional.
# If no params are required by the model, then it should be an empty list.
model_infer_params = []

# [Req] Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
infer_params = app_infer_params + model_infer_params

# moved/modified from batcher.py
def create_batch(batch_size, label, positions, response_dict, drug_smile, mutations, dataset_type=None, rseed=1):
    """Creates batch object"""

    # transform drug response matrix
    assert label in response_dict, f"key {label} not in dictionary"
    value_shape = response_dict[label].shape
    value = np.zeros((value_shape[0], value_shape[1], 1))
    value[ :, :, 0 ] = response_dict[label]

    # transpose dataframe
    drug_smile = np.transpose(drug_smile, (0, 2, 1)) 

    # create batch object
    ds = Batch(batch_size, value, drug_smile, mutations, positions)
    
    return ds

def initialize_parameters(default_model="tcnns_default_model.txt"):

    # Build benchmark object
    common = tcnns.tCNNS(
        filepath,
        default_model,
        "tensorflow",
        prog="twin Convolutional Neural Network for drugs in SMILES format (tCNNS)",
        desc="tCNNS drug response prediction model",
    )

    # Initialize parameters
    gParameters = candle.finalize_parameters(common)

    return gParameters

def load_graph(meta_file):
    """Creates new graph and session"""
    graph = tf.Graph()
    with graph.as_default():
        # Create session and load model
        sess = tf.Session()

        # Load meta file
        print("Loading meta graph from " + meta_file)
        saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
    return graph, sess, saver

def load_ckpt(ckpt, sess, saver):
    """Helper for loading weights"""
    # Load weights
    if ckpt is not None:
        print(f"Loading weights from {ckpt} folder...")
        saver.restore(sess, tf.train.latest_checkpoint(ckpt))

# [Req]
def run(params: Dict):
    """ Run model inference.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on test data according
            to the metrics_list.
    """ 

    args = candle.ArgumentStruct(**params)
    
    # ------------------------------------------------------
    # [Req] Create output dir
    # ------------------------------------------------------
    frm.create_outdir(outdir=params["infer_outdir"])

    # ------------------------------------------------------
    # [Req] Create data name for test set
    # ------------------------------------------------------
    #test_data_fname = frm.build_ml_data_name(params, stage="test")
    
    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    # load processed data and create batch object
    print("Loading data...")
    
    batch_size=1
    y_col_name=params["y_col_name"]
    
    if args.use_original_data:
        # load processed data
        drug_smile_dict = np.load(os.path.join(args.data_dir, args.data_subdir, args.drug_file), encoding="latin1", allow_pickle=True).item()
        drug_cell_dict = np.load(os.path.join(args.data_dir, args.data_subdir, args.response_file), encoding="latin1", allow_pickle=True).item()
        cell_mut_dict = np.load(os.path.join(args.data_dir, args.data_subdir, args.cell_file), encoding="latin1", allow_pickle=True).item()
        test_positions = np.load(os.path.join(args.data_dir, args.data_subdir, args.test_indices_file), encoding="latin1", allow_pickle=True).item()
        test = create_batch(batch_size, y_col_name, test_positions["positions"], drug_cell_dict, drug_smile_dict["canonical"], cell_mut_dict["cell_mut"])
    else:
        # load test data
        drug_smile_dict = np.load(Path(params["test_ml_data_dir"])/"test_drug_onehot_smiles.npy", encoding="latin1", allow_pickle=True).item()
        drug_cell_dict = np.load(Path(params["test_ml_data_dir"])/"test_drug_cell_interaction.npy", encoding="latin1", allow_pickle=True).item()
        cell_mut_dict = np.load(Path(params["test_ml_data_dir"])/"test_cell_mut_matrix.npy", encoding="latin1", allow_pickle=True).item()
        test = create_batch(batch_size, y_col_name, drug_cell_dict["positions"], drug_cell_dict, drug_smile_dict["canonical"], cell_mut_dict["cell_mut"])
 
    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Build model path
    modelpath = frm.build_model_path(params, model_dir=params["model_dir"]) # [Req]

    #y_col_name = args.label_name[0] # label

    # load model
    print("Loading trained model...")

    # Load metagraph and create session
    graph, sess, saver = load_graph(os.path.join(modelpath, args.model_weights_file))

    # Load checkpoint
    with graph.as_default():
        load_ckpt(modelpath, sess, saver)

        # run model to get predictions
        print("Obtainings predictions from trained model...")

        output_layer = graph.get_tensor_by_name("output_tensor:0")
        test_pred = []
        drug_id_list = []
        cell_id_list = []
        test_true = []
        for i in range(len(test.positions)):
            row = test.positions[i][0]
            col = test.positions[i][1]
            test_drug = np.array(test.drug[row])
            #drug_id_list.append(drug_cell_dict[params.drug_col_name][row])
            test_cell = np.array(test.cell[col])
            #cell_id_list.append(drug_cell_dict[params.canc_col_name][col])
            test_value = np.array(test.value[row, col])
            test_true.append(test_value[0])
        
            prediction = sess.run(output_layer, feed_dict={"Placeholder:0": np.reshape(test_drug,(1,test_drug.shape[0],test_drug.shape[1])),
                                                "Placeholder_1:0": np.reshape(test_cell, (1, test_cell.shape[0])), 
                                                "Placeholder_2:0": np.reshape(test_value, (1, test_value.shape[0])),
                                                "Placeholder_3:0": 1}) # keep_prob

            test_pred.append(prediction[0][0])
 
    """
    # save predictions to file
    print("Preparing predictions file...")
    # create prediction dataframe
    pred_col_name = y_col_name + ig.pred_col_name_suffix
    pred_df = pd.DataFrame(zip(drug_id_list,cell_id_list,value_list), columns=[ig.drug_col_name,ig.canc_col_name,y_col_name])
    # add prediction from model
    pred_df[pred_col_name] = test_predict
    if (y_col_name == "IC50") and (args.norm): # original data's normalized IC50
        # reverse normalization of true values
        pred_df[y_col_name] = pred_df[y_col_name].apply(lambda x: math.log(((1-x)/x)**-10))
        # reverse normalization of predicted values
        pred_df[pred_col_name] = pred_df[pred_col_name].apply(lambda x: math.log(((1-x)/x)**-10))
    pred_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    #r2 = improve_utils.r_square(pred_df[y_col_name], pred_df[pred_col_name])
    #print(f"R-square of test dataset: {np.round(r2, 5)}")
    """
    if (params["y_col_name"].lower() == "ic50") and (args.norm): # original data's normalized IC50
        # reverse normalization of true values
        test_true = test_true.apply(lambda x: math.log(((1-x)/x)**-10))
        # reverse normalization of predicted values
        test_pred = test_pred.apply(lambda x: math.log(((1-x)/x)**-10))
        
    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        params,
        y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"]
    )
    
    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    test_scores = frm.compute_performace_scores(
        params,
        y_true=test_true, y_pred=test_pred, stage="test",
        outdir=params["infer_outdir"], metrics=metrics_list
    )
    
    return test_scores

# [Req]
def main(args):
    start = time.time()
    # [Req]
    additional_definitions = preprocess_params + train_params + infer_params
    params = frm.initialize_parameters(
        filepath,
        default_model="tcnns_csa_params.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    test_scores = run(params)
    print("\nFinished model inference.")
    end = time.time()
    print("Total runtime: {}".format(end-start))

# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])

import os
from pathlib import Path
import candle
import tensorflow as tf
from batcher import *
import tcnns
import numpy as np
import json
import pandas as pd
import math
import improve_utils
from improve_utils import improve_globals as ig
import time

# get file path of script
file_path = os.path.dirname(os.path.realpath(__file__))

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
        file_path,
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

def run(gParameters): 

    args = candle.ArgumentStruct(**gParameters)

    y_col_name = args.label_name[0] # label

    # load processed data
    print("Loading data...")
    if args.use_original_data:
        # load processed data
        drug_smile_dict = np.load(os.path.join(args.data_dir, args.data_subdir, args.drug_file), encoding="latin1", allow_pickle=True).item()
        drug_cell_dict = np.load(os.path.join(args.data_dir, args.data_subdir, args.response_file), encoding="latin1", allow_pickle=True).item()
        cell_mut_dict = np.load(os.path.join(args.data_dir, args.data_subdir, args.cell_file), encoding="latin1", allow_pickle=True).item()
        test_positions = np.load(os.path.join(args.data_dir, args.data_subdir, args.test_indices_file), encoding="latin1", allow_pickle=True).item()
    
    batch_size = 1

    # restructure data for model 
    # create batch object
    test = create_batch(batch_size, y_col_name, test_positions["positions"], drug_cell_dict, drug_smile_dict["canonical"], cell_mut_dict["cell_mut"])

    # load model
    print("Loading trained model...")

    # Load metagraph and create session
    graph, sess, saver = load_graph(os.path.join(args.ckpt_directory, args.model_weights_file))

    # Load checkpoint
    with graph.as_default():
        load_ckpt(args.ckpt_directory, sess, saver)

        # run model to get predictions
        print("Obtainings predictions from trained model...")

        output_layer = graph.get_tensor_by_name("output_tensor:0")
        test_predict = []
        drug_id_list = []
        cell_id_list = []
        value_list = []
        for i in range(len(test.positions)):
            row = test.positions[i][0]
            col = test.positions[i][1]
            test_drug = np.array(test.drug[row])
            drug_id_list.append(drug_cell_dict[args.drug_id][row])
            test_cell = np.array(test.cell[col])
            cell_id_list.append(drug_cell_dict[args.cell_id][col])
            test_value = np.array(test.value[row, col])
            value_list.append(test_value[0])
        
            prediction = sess.run(output_layer, feed_dict={"Placeholder:0": np.reshape(test_drug,(1,test_drug.shape[0],test_drug.shape[1])),
                                                "Placeholder_1:0": np.reshape(test_cell, (1, test_cell.shape[0])), 
                                                "Placeholder_2:0": np.reshape(test_value, (1, test_value.shape[0])),
                                                "Placeholder_3:0": 1}) # keep_prob

            test_predict.append(prediction[0][0])
 
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

    r2 = improve_utils.r_square(pred_df[y_col_name], pred_df[pred_col_name])
    print(f"R-square of test dataset: {np.round(r2, 5)}")

def main():
    start = time.time()
    gParameters = initialize_parameters()
    run(gParameters)
    end = time.time()
    print("Total runtime: {}".format(end-start))
    print("Finished.")

if __name__ == "__main__":
    main()

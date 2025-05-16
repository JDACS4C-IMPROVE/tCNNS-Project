import os
from pathlib import Path
import tensorflow as tf
from batcher import Batch
import numpy as np
import pandas as pd
import math
import sys
from typing import Dict

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
from improvelib.utils import str2bool
import improvelib.utils as frm

# Model-specific imports
from model_params_def import infer_params
from batcher import Batch

# [Req]
filepath = Path(__file__).resolve().parent 

# Moved/modified from batcher.py
def create_batch(batch_size, label, positions, response_dict, drug_smile, mutations, dataset_type=None, rseed=1):
    """Creates batch object"""

    # Transform drug response matrix
    assert label in response_dict, f"key {label} not in dictionary"
    value_shape = response_dict[label].shape
    value = np.zeros((value_shape[0], value_shape[1], 1))
    #value[ :, :, 0 ] = response_dict[label]
    value[ :, :,  ] = response_dict[label]

    # Transpose dataframe
    drug_smile = np.transpose(drug_smile, (0, 2, 1)) 

    # Create batch object
    ds = Batch(batch_size, value, drug_smile, mutations, positions)
    
    return ds

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
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on test data.
    """
    # ------------------------------------------------------
    # [Req] Check for GPU availability
    # ------------------------------------------------------
    if tf.test.gpu_device_name():
        if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
            print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
        print("GPU Device:{}".format(tf.test.gpu_device_name()))
    else:
        print("GPU not available")
    
    # ------------------------------------------------------
    # [Req] Create output dir
    # ------------------------------------------------------
    frm.create_outdir(outdir=params["output_dir"])

    # ------------------------------------------------------
    # [Req] Create data name for test set
    # ------------------------------------------------------
    
    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    # Load processed data and create batch object
    print("Loading data...")
        
    if params["use_original_data"]:
        # Load processed data
        drug_smile_dict = np.load(os.path.join(params["data_dir"], params["data_subdir"], params["drug_file"]), encoding="latin1", allow_pickle=True).item()
        drug_cell_dict = np.load(os.path.join(params["data_dir"], params["data_subdir"], params["response_file"]), encoding="latin1", allow_pickle=True).item()
        cell_mut_dict = np.load(os.path.join(params["data_dir"], params["data_subdir"], params["cell_file"]), encoding="latin1", allow_pickle=True).item()
        test_positions = np.load(os.path.join(params["data_dir"], params["data_subdir"], params["test_indices_file"]), encoding="latin1", allow_pickle=True).item()
        test = create_batch(params["infer_batch"], params["y_col_name"], test_positions["positions"], drug_cell_dict, drug_smile_dict["canonical"], cell_mut_dict["cell_mut"])
    else:
        # Load test data
        drug_smile_dict = np.load(Path(params["input_data_dir"])/"test_drug_onehot_smiles.npy", encoding="latin1", allow_pickle=True).item()
        drug_cell_dict = np.load(Path(params["input_data_dir"])/"test_drug_cell_interaction.npy", encoding="latin1", allow_pickle=True).item()
        cell_mut_dict = np.load(Path(params["input_data_dir"])/"test_cell_mut_matrix.npy", encoding="latin1", allow_pickle=True).item()
        test = create_batch(params["infer_batch"], params["y_col_name"], drug_cell_dict["positions"], drug_cell_dict, drug_smile_dict["canonical"], cell_mut_dict["cell_mut"])
 
    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # [Req] Build model path
    modelpath = frm.build_model_path(model_file_name=params["model_file_name"], model_file_format=params["model_file_format"], model_dir=params["input_model_dir"])
    
    # Load model
    print("Loading trained model...")

    # Load metagraph and create session
    graph, sess, saver = load_graph(str(modelpath))

    # Load checkpoint
    with graph.as_default():
        ckptpath = os.path.join(params["input_model_dir"], "model")
        load_ckpt(ckptpath, sess, saver)

        # Run model to get predictions
        print("Obtainings predictions from trained model...")

        output_layer = graph.get_tensor_by_name("output_tensor:0")
        test_pred = []
        test_true = []

        for i in range(len(test.positions)):
            row = test.positions[i][0]
            col = test.positions[i][1]
            tidx = test.positions[i][2]
            test_drug = np.array(test.drug[row])
            test_cell = np.array(test.cell[col])
            test_value = np.array(test.value[row, tidx])
            test_true.append(drug_cell_dict["response_values"][i])
        
            prediction = sess.run(output_layer, feed_dict={"Placeholder:0": np.reshape(test_drug,(1,test_drug.shape[0],test_drug.shape[1])),
                                                "Placeholder_1:0": np.reshape(test_cell, (1, test_cell.shape[0])), 
                                                "Placeholder_2:0": np.reshape(test_value, (1, test_value.shape[0])),
                                                "Placeholder_3:0": 1}) # keep_prob

            test_pred.append(prediction[0][0])
 
    if (params["y_col_name"].lower() == "ic50") and (params["norm"]): # original data's normalized IC50
        # Reverse normalization of true values
        test_true = test_true.apply(lambda x: math.log(((1-x)/x)**-10))
        # Reverse normalization of predicted values
        test_pred = test_pred.apply(lambda x: math.log(((1-x)/x)**-10))
        
    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=test_true, 
        y_pred=test_pred, 
        stage="test",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_data_dir"]
    )
    
    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    if params["calc_infer_scores"]:
        test_scores = frm.compute_performance_scores(
            y_true=test_true, 
            y_pred=test_pred, 
            stage="test",
            metric_type=params["metric_type"],
            output_dir=params["output_dir"]
        )
 
    return True

# [Req]
def main(args):
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="tcnns_params.ini",
        additional_definitions=infer_params,
        required=None,
    )
    timer_infer = frm.Timer()  
    status = run(params)
    timer_infer.save_timer(dir_to_save=params["output_dir"], 
                           filename='runtime_infer.json', 
                           extra_dict={"stage": "infer"})
    print("\nFinished model inference.")

# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])

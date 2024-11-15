import os
from pathlib import Path
import tensorflow as tf
from batcher import *
import numpy as np
import json
from sklearn.metrics import mean_squared_error
from scipy import stats
import pandas as pd
import math
import time
import subprocess
from typing import Dict
import sys

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from model_params_def import train_params

# [Req]
filepath = Path(__file__).resolve().parent

def weight_variable(shape, std_dev, rseed):
    initial = tf.truncated_normal(shape, stddev=std_dev, seed=rseed)
    return tf.Variable(initial)

def bias_variable(shape, bias_constant):
    initial = tf.constant(bias_constant, shape=shape)
    return tf.Variable(initial) 

def conv1d(x, W, conv_stride):
    return tf.nn.conv1d(x, W, conv_stride, padding='SAME')

def max_pool_1d(x, kernel_shape, strides, padding='SAME'):
    return tf.nn.pool(x, kernel_shape, 'MAX', padding=padding, strides=strides)

def R2(real, predicted):
    label_mean = tf.reduce_mean(real, axis=0)
    total_sum_of_square = tf.reduce_sum(tf.square(real - label_mean), axis=0)
    residual_sum_of_square = tf.reduce_sum(tf.square(real - predicted), axis=0)
    r2 = 1 - residual_sum_of_square / total_sum_of_square
    return r2

def Pearson(a, b):
    real = tf.squeeze(a)
    pred = tf.squeeze(b)
    real_new = real - tf.reduce_mean(real)
    pred_new = pred - tf.reduce_mean(real)
    up = tf.reduce_mean(tf.multiply(real_new, pred_new))
    real_var = tf.reduce_mean(tf.multiply(real_new, real_new))
    pred_var = tf.reduce_mean(tf.multiply(pred_new, pred_new))
    down = tf.multiply(tf.sqrt(real_var), tf.sqrt(pred_var))
    return tf.div(up, down)

def Spearman(real, predicted):
    rs = tf.py_function(stats.spearmanr, [tf.cast(predicted, tf.float32), 
                       tf.cast(real, tf.float32)], Tout = tf.float32)
    return rs

# moved from batcher.py
def load_data(batch_size, label_list, positions, response_dict, smiles_canonical, mutations, train_size, val_size):
    """Splits data into train, test, and validation datasets and creates batch objects for each dataset"""
    size = positions.shape[0]
    assert 0.0 <= train_size <= 1.0, "Training set size must be between 0.0 and 1.0."
    assert train_size <= val_size <= 1.0, "Validation set size must be between train_size and 1.0."
    len1 = int(size * train_size)
    len2 = int(size * val_size)

    train_pos = positions[0 : len1]
    valid_pos = positions[len1 : len2]
    test_pos = positions[len2 : ]

    value_shape = response_dict[label_list[0]].shape
    value = np.zeros((value_shape[0], value_shape[1], len(label_list)))

    for i in range(len(label_list)):
        value[ :, :, i ] = response_dict[label_list[i]]
    
    drug_smile = smiles_canonical

    train = Batch(batch_size, value, drug_smile, mutations, train_pos)
    valid = Batch(batch_size, value, drug_smile, mutations, valid_pos)
    test = Batch(batch_size, value, drug_smile, mutations, test_pos)
    
    return train, valid, test

# moved/modified from batcher.py
def create_batch(batch_size, label, positions, response_dict, drug_smile, mutations, dataset_type=None, rseed=1):
    if dataset_type == "train":
        np.random.seed(rseed)
        np.random.shuffle(positions)

    # check batch order of train? same drugs in each batch?

    # transform drug response matrix
    assert label in response_dict, f"key {label} not in dictionary"
    value_shape = response_dict[label].shape
    value = np.zeros((value_shape[0], value_shape[1], 1))
    #value[ :, :, 0 ] = response_dict[label]
    value[ :, :, ] = response_dict[label]

    # transpose dataframe
    drug_smile = np.transpose(drug_smile, (0, 2, 1)) 

    # create batch object
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

"""
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
"""

# [Req]
def run(params: Dict):
    """ Run model training.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on validation data.
    """
    # ------------------------------------------------------
    # [Req] Create output dir and build model path
    # ------------------------------------------------------
    # Create output dir for trained model, val set predictions, val set
    # performance scores
    frm.create_outdir(outdir=params["output_dir"])

    # Build model path
    #modelpath = frm.build_model_path(model_dir=params["output_dir"])
    modelpath = os.path.join(params["output_dir"], params["model_file_name"]) #TODO model_sub_dir

    # check for GPU
    if tf.test.gpu_device_name():
        if os.getenv("CUDA_VISIBLE_DEVICES") is not None:
            print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
        print("GPU Device:{}".format(tf.test.gpu_device_name()))
    else:
        print("GPU not available")
    
    #if params["use_original_data"]:
        # get data from server if processed original data is not available
        #candle.file_utils.get_file(args.processed_data, f"{args.data_url}/{args.processed_data}", cache_subdir = args.cache_subdir)

    # check files in data processed folder
    #proc = subprocess.Popen([f"ls {args.data_dir}/{args.data_subdir}/*"], stdout=subprocess.PIPE, shell=True)   
    #(out, err) = proc.communicate()
    #print("List of files in processed data folder", out.decode('utf-8'))
    
    # ------------------------------------------------------
    # [Req] Create data names for train and val sets
    # ------------------------------------------------------
    #train_data_fname = frm.build_ml_data_name(params, stage="train")
    #val_data_fname = frm.build_ml_data_name(params, stage="val")
    #print(train_data_fname)
    #print(val_data_fname)
    
    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    if params["use_original_data"]:
        drug_smile_dict = np.load(os.path.join(params["data_dir"], params["data_subdir"], params["drug_file"]), encoding="latin1", allow_pickle=True).item()
        drug_cell_dict = np.load(os.path.join(params["data_dir"], params["data_subdir"], params["response_file"]), encoding="latin1", allow_pickle=True).item()
        cell_mut_dict = np.load(os.path.join(params["data_dir"], params["data_subdir"], params["cell_file"]), encoding="latin1", allow_pickle=True).item()
    else:
        # train data
        drug_smile_dict = np.load(Path(params["input_dir"])/"train_drug_onehot_smiles.npy", encoding="latin1", allow_pickle=True).item()
        drug_cell_dict = np.load(Path(params["input_dir"])/"train_drug_cell_interaction.npy", encoding="latin1", allow_pickle=True).item()
        cell_mut_dict = np.load(Path(params["input_dir"])/"train_cell_mut_matrix.npy", encoding="latin1", allow_pickle=True).item()
        # val data
        vl_drug_smile_dict = np.load(Path(params["input_dir"])/"val_drug_onehot_smiles.npy", encoding="latin1", allow_pickle=True).item()
        vl_drug_cell_dict = np.load(Path(params["input_dir"])/"val_drug_cell_interaction.npy", encoding="latin1", allow_pickle=True).item()
        vl_cell_mut_dict = np.load(Path(params["input_dir"])/"val_cell_mut_matrix.npy", encoding="latin1", allow_pickle=True).item()

    # define variables
    c_chars = drug_smile_dict["c_chars"]
    canonical = drug_smile_dict["canonical"]
    canonical = np.transpose(canonical, (0, 2, 1))
    mut_names = cell_mut_dict["mut_names"]
    cell_mut = cell_mut_dict["cell_mut"]
    all_positions = drug_cell_dict["positions"] # array of zipped object
    all_positions = np.array(list(all_positions.tolist()))
    np.random.seed(params["rng_seed"])
    np.random.shuffle(all_positions)
    length_smiles = len(canonical[0]) # length of smiles
    num_cell_features = len(mut_names) # number of mutations
    num_chars_smiles = len(c_chars) # number of characters in smiles
    print("Length of SMILES string: {}".format(length_smiles))
    print("Number of mutations: {}".format(num_cell_features))
    print("Number of unique characters in SMILES string: {}".format(num_chars_smiles))
       
    # ------------------------------------------------------
    # Prepare, train, and save model
    # ------------------------------------------------------

    # define model
    drug = tf.placeholder(tf.float32, shape=[None, length_smiles, num_chars_smiles])
    cell = tf.placeholder(tf.float32, shape=[None, num_cell_features])
    scores = tf.placeholder(tf.float32, shape=[None, 1])
    keep_prob = tf.placeholder(tf.float32)

    # define drug convolutional layers
    for i in range(0, len(params["drug_conv_out"])):
        if i == 0:
            drug_conv_out = params["drug_conv_out"][i] 
            drug_conv_pool = params["drug_pool"][i]
            drug_conv_w = weight_variable([params["drug_conv_width"][i], num_chars_smiles, drug_conv_out], params["std_dev"], params["rng_seed"])
            drug_conv_b = bias_variable([drug_conv_out], params["bias_constant"])
            drug_conv_h = tf.nn.relu(conv1d(drug, drug_conv_w, params["conv_stride"]) + drug_conv_b)
            drug_conv_p = max_pool_1d(drug_conv_h, [drug_conv_pool], [drug_conv_pool])
        else:
            drug_conv_out = params["drug_conv_out"][i]
            drug_conv_pool = params["drug_pool"][i]
            drug_conv_w = weight_variable([params["drug_conv_width"][i], params["drug_conv_out"][i-1], drug_conv_out], params["std_dev"], params["rng_seed"])
            drug_conv_b = bias_variable([drug_conv_out], params["bias_constant"])
            drug_conv_h = tf.nn.relu(conv1d(drug_conv_p, drug_conv_w, params["conv_stride"]) + drug_conv_b)
            drug_conv_p = max_pool_1d(drug_conv_h, [drug_conv_pool], [drug_conv_pool])

    # define cell convolutional layers
    for i in range(0, len(params["cell_conv_out"])):
        if i == 0:
            cell_conv_out = params["cell_conv_out"][i]
            cell_conv_pool = params["cell_pool"][i]
            cell_tensor = tf.expand_dims(cell, 2)
            cell_conv_w = weight_variable([params["cell_conv_width"][i], 1, cell_conv_out], params["std_dev"], params["rng_seed"])
            cell_conv_b = weight_variable([cell_conv_out], params["bias_constant"], params["rng_seed"])
            cell_conv_h = tf.nn.relu(conv1d(cell_tensor, cell_conv_w, params["conv_stride"]) + cell_conv_b)
            cell_conv_p = max_pool_1d(cell_conv_h, [cell_conv_pool], [cell_conv_pool])
        else: 
            cell_conv_out = params["cell_conv_out"][i]
            cell_conv_pool = params["cell_pool"][i]
            cell_tensor = tf.expand_dims(cell, 2)
            cell_conv_w = weight_variable([params["cell_conv_width"][i], params["cell_conv_out"][i-1], cell_conv_out], params["std_dev"], params["rng_seed"])
            cell_conv_b = bias_variable([cell_conv_out], params["bias_constant"])
            cell_conv_h = tf.nn.relu(conv1d(cell_conv_p, cell_conv_w, params["conv_stride"]) + cell_conv_b)
            cell_conv_p = max_pool_1d(cell_conv_h, [cell_conv_pool], [cell_conv_pool])

    # merge drug and cell convolutional layers
    conv_merge = tf.concat([drug_conv_p, cell_conv_p], 1)
    # reshape layer for fully connected layers
    shape = conv_merge.get_shape().as_list()
    conv_flat = tf.reshape(conv_merge, [-1, shape[1] * shape[2]])

    # define fully connected layers
    for i in range(0, len(params["dense"])):
        if i == 0:
            fc_w = weight_variable([shape[1] * shape[2], params["dense"][i]], params["std_dev"], params["rng_seed"])
            fc_b = bias_variable([params["dense"][i]], params["bias_constant"])
            fc_h = tf.nn.relu(tf.matmul(conv_flat, fc_w) + fc_b)
            fc_drop = tf.nn.dropout(fc_h, keep_prob)
        elif i == (len(params["dense"]) - 1): 
            fc_w = weight_variable([params["dense"][i], 1], params["std_dev"], params["rng_seed"])
            fc_b = weight_variable([1], params["std_dev"], params["rng_seed"])
        else:
            fc_w = weight_variable([params["dense"][i], params["dense"][i]], params["std_dev"], params["rng_seed"])
            fc_b = bias_variable([params["dense"][i]], params["bias_constant"])
            fc_h = tf.nn.relu(tf.matmul(fc_drop, fc_w) + fc_b)
            fc_drop = tf.nn.dropout(fc_h, keep_prob)

    if params["out_activation"] == "sigmoid":
        # use sigmoid function on output layer; recommended for original data's normalized IC50
        y_conv = tf.nn.sigmoid(tf.matmul(fc_drop, fc_w) + fc_b, name="output_tensor")
    else:
        y_conv = tf.nn.xw_plus_b(fc_drop, fc_w, fc_b, name="output_tensor")

    # define loss
    loss = tf.losses.mean_squared_error(scores, y_conv)
    # define optimizer
    train_step = tf.train.AdamOptimizer(params["learning_rate"]).minimize(loss)

    # define metrics
    r_square = R2(scores, y_conv)
    pearson = Pearson(scores, y_conv)
    rmse = tf.sqrt(loss)
    spearman = Spearman(scores, y_conv)

    # if using original data:
    if params["use_original_data"]:
        # split data into train, valid, and test datasets
        train, valid, test = load_data(params["batch_size"], params["label_name"], all_positions, drug_cell_dict, canonical, cell_mut, params["train_size"], params["val_size"])
        # save test positions for inference
        save_dict = {}
        save_dict["positions"] = test.positions
        np.save(os.path.join(params["data_dir"], params["data_subdir"], "test_positions.npy"), save_dict)
        print("Saving test data indices for inference.")
    else:
           # create train, valid, and test batch objects
        train = create_batch(params["batch_size"], params["y_col_name"], drug_cell_dict["positions"], drug_cell_dict, drug_smile_dict["canonical"], cell_mut_dict["cell_mut"], dataset_type="train", rseed=params["rng_seed"])
        valid = create_batch(params["batch_size"], params["y_col_name"], vl_drug_cell_dict["positions"], vl_drug_cell_dict, vl_drug_smile_dict["canonical"], vl_cell_mut_dict["cell_mut"])
        
    # initialize saver object
    saver = tf.train.Saver(var_list=tf.trainable_variables())

    # train model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)

        #test_values, test_drugs, test_cells = test.whole_batch()
        valid_values, valid_drugs, valid_cells = valid.whole_batch()
        epoch = 0
        best_epoch = 0
        min_loss = params["min_loss"]
        count = 0
        epoch_time = []
        val_scores = {}
        # option only runs early stopping
        if params["epochs"] == 0:
            while count < params["es_epochs"]: 
                epoch_start_time = time.time()
                train.reset()
                step = 0
                while(train.available()):
                    real_values, drug_smiles, cell_muts = train.mini_batch()
                    train_step.run(feed_dict={drug:drug_smiles, cell:cell_muts, scores:real_values, keep_prob:params["dropout"]})
                    step += 1
                valid_loss, valid_r2, valid_pcc, valid_rmse, valid_scc = sess.run([loss, r_square, pearson, rmse, spearman], feed_dict={drug:valid_drugs, cell:valid_cells, scores:valid_values, keep_prob:1})
                print("epoch: %d, loss: %g r2: %g pearson: %g rmse: %g, spearman: %g" % (epoch, valid_loss, valid_r2, valid_pcc, valid_rmse, valid_scc))
                if valid_loss < min_loss:
                    print("epoch with lowest val_loss: %d, loss: %g" % (epoch, valid_loss))
                    best_epoch = epoch
                    # save scores associated with lowest validation loss
                    val_scores = {"val_loss": float(valid_loss), "r2": float(valid_r2), "pcc": float(valid_pcc), "scc": float(valid_scc), "rmse": float(valid_rmse)}
                    #os.system("rm {}/*".format(os.path.join(args.output_dir, args.ckpt_directory)))
                    #saver.save(sess, os.path.join(args.output_dir, args.ckpt_directory, "result.ckpt"))
                    os.system("rm {}/*".format(modelpath))
                    saver.save(sess, os.path.join(modelpath, "result.ckpt"))
                    print("Model saved!")
                    min_loss = valid_loss
                    count = 0
                else:
                    count = count + 1
                epoch += 1
                epoch_end_time = time.time()
                epoch_time.append(epoch_end_time - epoch_start_time)
        else:
            # option runs model for x epochs (no early stopping)
            if params["es_epochs"] == 0:
                if params["epochs"] == 0:
                    print("Please specify number of epochs.")
                else:
                    for epoch in range(params["epochs"]):
                        epoch_start_time = time.time()
                        train.reset()
                        step = 0
                        while(train.available()):
                            real_values, drug_smiles, cell_muts = train.mini_batch()
                            train_step.run(feed_dict={drug:drug_smiles, cell:cell_muts, scores:real_values, keep_prob:params["dropout"]})
                            step += 1
                        valid_loss, valid_r2, valid_pcc, valid_rmse, valid_scc = sess.run([loss, r_square, pearson, rmse, spearman], feed_dict={drug:valid_drugs, cell:valid_cells, scores:valid_values, keep_prob:1})
                        print("epoch: %d, loss: %g r2: %g pearson: %g rmse: %g, spearman: %g" % (epoch, valid_loss, valid_r2, valid_pcc, valid_rmse, valid_scc))
                        if valid_loss < min_loss:
                            print("epoch with lowest val_loss: %d, loss: %g" % (epoch, valid_loss))
                            best_epoch = epoch
                            # save scores associated with lowest validation loss
                            val_scores = {"val_loss": float(valid_loss), "r2": float(valid_r2), "pcc": float(valid_pcc), "scc": float(valid_scc), "rmse": float(valid_rmse)}
                            #os.system("rm {}/*".format(os.path.join(args.output_dir, args.ckpt_directory)))
                            #saver.save(sess, os.path.join(args.output_dir, args.ckpt_directory, "result.ckpt"))
                            os.system("rm {}/*".format(modelpath))
                            saver.save(sess, os.path.join(modelpath, "result.ckpt"))
                            print("Model saved!")
                            min_loss = valid_loss
                        epoch_end_time = time.time()
                        epoch_time.append(epoch_end_time - epoch_start_time)    
            else:
                # option runs model for x epochs and uses early stopping
                while count < params["es_epochs"]:
                    epoch_start_time = time.time()
                    train.reset()
                    step = 0
                    while(train.available()):
                        real_values, drug_smiles, cell_muts = train.mini_batch()
                        train_step.run(feed_dict={drug:drug_smiles, cell:cell_muts, scores:real_values, keep_prob:params["dropout"]})
                        step += 1
                    valid_loss, valid_r2, valid_pcc, valid_rmse, valid_scc = sess.run([loss, r_square, pearson, rmse, spearman], feed_dict={drug:valid_drugs, cell:valid_cells, scores:valid_values, keep_prob:1})
                    print("epoch: %d, loss: %g r2: %g pearson: %g rmse: %g, spearman: %g" % (epoch, valid_loss, valid_r2, valid_pcc, valid_rmse, valid_scc))
                    if valid_loss < min_loss:
                        print("epoch with lowest val_loss: %d, loss: %g" % (epoch, valid_loss))
                        best_epoch = epoch
                        # save scores associated with lowest validation loss
                        val_scores = {"val_loss": float(valid_loss), "r2": float(valid_r2), "pcc": float(valid_pcc), "scc": float(valid_scc), "rmse": float(valid_rmse)}
                        #os.system("rm {}/*".format(os.path.join(args.output_dir, args.ckpt_directory)))
                        #saver.save(sess, os.path.join(args.output_dir, args.ckpt_directory, "result.ckpt"))
                        os.system("rm {}/*".format(modelpath))
                        saver.save(sess, os.path.join(modelpath, "result.ckpt"))
                        print("Model saved!")
                        min_loss = valid_loss
                        count = 0
                    else:
                        count = count + 1
                    epoch += 1
                    epoch_end_time = time.time()
                    epoch_time.append(epoch_end_time - epoch_start_time)
                    if epoch == params["epochs"]:
                        break

        if params["epochs"]>0 and params["es_epochs"]>0:
            print(f"Total number of epochs: {epoch}.")
        else:
            print(f"Total number of epochs: {epoch+1}.")
        print(f"Best epoch with lowest val_loss: {best_epoch}.")
        print(f"Runtime for first epoch: {epoch_time[0]}")
        print(f"Average runtime per epoch: {sum(epoch_time)/len(epoch_time)}")
    
    """
    # Supervisor HPO
    if len(val_scores) > 0:
        print("\nIMPROVE_RESULT val_loss:\t{}\n".format(val_scores["val_loss"]))
        with open(Path(params["output_dir"]) / "scores.json", "w", encoding="utf-8") as f:
            json.dump(val_scores, f, ensure_ascii=False, indent=4)
    else:
        print("The val_loss did not improve from the min_loss after training. Results and model not saved.")
    """
    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Load the best saved model (as determined based on val data)
    # Load metagraph and create session
    print(f"This is the path to model: {modelpath}")
    graph, sess, saver = load_graph(os.path.join(modelpath, "result.ckpt.meta"))

    # Load checkpoint
    with graph.as_default():
        load_ckpt(modelpath, sess, saver)

        # run model to get predictions
        print("Obtainings predictions from trained model...")

        output_layer = graph.get_tensor_by_name("output_tensor:0")
        val_pred = []
        drug_id_list = []
        cell_id_list = []
        val_true = []
        for i in range(len(valid.positions)):
            row = valid.positions[i][0] # didx
            col = valid.positions[i][1] # cidx
            tidx = valid.positions[i][2]
            valid_drug = np.array(valid.drug[row])
            #drug_id_list.append(drug_cell_dict[params["drug_col_name"]][row])
            valid_cell = np.array(valid.cell[col])
            #cell_id_list.append(drug_cell_dict[params["canc_col_name"]][col])
            #valid_value = np.array(valid.value[row, col])
            valid_value = np.array(valid.value[row, tidx])
            #print(valid_value)
            #val_true.append(valid_value[0])
            val_true.append(vl_drug_cell_dict["response_values"][i])
            #print(vl_drug_cell_dict["response_values"][i])
        
            prediction = sess.run(output_layer, feed_dict={"Placeholder:0": np.reshape(valid_drug,(1,valid_drug.shape[0],valid_drug.shape[1])),
                                                "Placeholder_1:0": np.reshape(valid_cell, (1, valid_cell.shape[0])), 
                                                "Placeholder_2:0": np.reshape(valid_value, (1, valid_value.shape[0])),
                                                "Placeholder_3:0": 1}) # keep_prob

            val_pred.append(prediction[0][0])
            
        # reverse normalization if using IC50
        if params["y_col_name"].lower() == "ic50":
            val_true = val_true.apply(lambda x: math.log(((1-x)/x)**-10))
            val_pred = val_pred.apply(lambda x: math.log(((1-x)/x)**-10))

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=val_true,
        y_pred=val_pred,
        stage="val",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_dir"]
    )
    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performance_scores(
        y_true=val_true,
        y_pred=val_pred,
        stage="val",
        metric_type=params["metric_type"],
        output_dir=params["output_dir"]
    )
        
    return val_scores
    
def initialize_parameters():
    """This initialize_parameters() is define this way to support Supervisor
    workflows such as HPO.

    Returns:
        dict: dict of IMPROVE/CANDLE parameters and parsed values.
    """
    # [Req] Initialize parameters
    cfg = DRPTrainConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="tcnns_params.txt",
        additional_definitions=train_params)
    return params


# [Req]
def main(args):
    start = time.time()
    # [Req]
    params = initialize_parameters()
    val_scores = run(params)
    print("\nFinished model training.")
    end = time.time()
    print("Total runtime: {}".format(end-start))

# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])

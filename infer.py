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

# get file path of script
file_path = os.path.dirname(os.path.realpath(__file__))

def load_data(batch_size, label_list, positions, response_dict, smiles_canonical, mutations):

    test_pos = positions # indices of drugs and cells

    # get drug response values per drug and cell pair
    value_shape = response_dict[label_list[0]].shape
    value = np.zeros((value_shape[0], value_shape[1], len(label_list)))

    for i in range(len(label_list)):
        key_name = label_list[i]
        assert key_name in response_dict, f"key {key_name} not in dictionary"
        value[ :, :, i ] = response_dict[label_list[i]]
    
    drug_smile = smiles_canonical

    # create batch object
    test = Batch(batch_size, value, drug_smile, mutations, test_pos)
    
    return test

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
        print('Loading meta graph from ' + meta_file)
        saver = tf.train.import_meta_graph(meta_file, clear_devices=True)
    return graph, sess, saver

def load_ckpt(ckpt, sess, saver):
    """Helper for loading weights"""
    # Load weights
    if ckpt is not None:
        print('Loading weights from ' + ckpt)
        saver.restore(sess, tf.train.latest_checkpoint(ckpt))


def run(gParameters): 

    args = candle.ArgumentStruct(**gParameters)

    # load processed data  TODO change arguments for inference
    print("Loading data...")
    drug_smile_dict = np.load(os.path.join(args.data_dir, args.data_subdir, args.drug_file), encoding="latin1", allow_pickle=True).item()
    drug_cell_dict = np.load(os.path.join(args.data_dir, args.data_subdir, args.response_file), encoding="latin1", allow_pickle=True).item()
    cell_mut_dict = np.load(os.path.join(args.data_dir, args.data_subdir, args.cell_file), encoding="latin1", allow_pickle=True).item()

    # define variables
    c_chars = drug_smile_dict["c_chars"]
    drug_names = drug_smile_dict["drug_names"]
    drug_cids = drug_smile_dict["drug_cids"]
    canonical = drug_smile_dict["canonical"]
    canonical = np.transpose(canonical, (0, 2, 1))
    cell_names = cell_mut_dict["cell_names"]
    mut_names = cell_mut_dict["mut_names"]
    cell_mut = cell_mut_dict["cell_mut"]
    cell_ids = drug_cell_dict["cell_ids"]
    all_positions = drug_cell_dict["positions"]
    all_positions = np.array(list(all_positions.tolist()))
    length_smiles = len(canonical[0])
    num_cell_features = len(mut_names)
    num_chars_smiles = len(c_chars)

    batch_size = 1

    # restructure data for model 
    test = load_data(batch_size, args.label_name, all_positions, drug_cell_dict, canonical, cell_mut)
    test_values, test_drugs, test_cells = test.whole_batch()

    # load model
    print("Loading trained model...")

    # Load metagraph and create session
    graph, sess, saver = load_graph(os.path.join(args.ckpt_directory, args.model_weights_file))

    # Load checkpoint
    with graph.as_default():
        load_ckpt(args.ckpt_directory, sess, saver)
    
    #with tf.Session() as sess:
        #test_predict = []
    #sess = tf.Session()    
        # load meta graph and restore weights
        #saver = tf.train.import_meta_graph(os.path.join(args.ckpt_directory, args.model_weights_file))
        #print(args.ckpt_directory)
        #saver.restore(sess,tf.train.latest_checkpoint(args.ckpt_directory))

    # define model placeholders
    #drug = tf.placeholder(tf.float32, shape=[None, length_smiles, num_chars_smiles])
    #cell = tf.placeholder(tf.float32, shape=[None, num_cell_features])
    #scores = tf.placeholder(tf.float32, shape=[None, 1])
    #keep_prob = tf.placeholder(tf.float32)

    # run model to get predictions
        print("Obtainings predictions from trained model...")
        #graph = tf.get_default_graph()
    #print(graph)
        print([tensor.name for tensor in tf.get_default_graph().as_graph_def().node])
        output_layer = graph.get_tensor_by_name("Sigmoid:0")


        test_predict = []
        for i in range(len(test.positions)):
            row = test.positions[i][0]
            col = test.positions[i][1]
            test_drug = np.array(test.drug[row])
            test_cell = np.array(test.cell[col])
            test_value = np.array(test.value[row, col])
        
            prediction = sess.run(output_layer, feed_dict={"Placeholder:0": np.reshape(test_drug,(1,test_drug.shape[0],test_drug.shape[1])),
                                                "Placeholder_1:0": np.reshape(test_cell, (1, test_cell.shape[0])), 
                                                "Placeholder_2:0": np.reshape(test_value, (1, test_value.shape[0])), # scores
                                                "Placeholder_3:0": 1}) # keep_prob
            #print(prediction[0])
            test_predict.append(prediction[0][0])

    #test_predict = sess.run(feed_dict={drug:test_drugs, cell:test_cells, scores:test_values, keep_prob:1})
 
    # save predictions to file
    print("Preparing predictions file...")
    # get drug names and indices
    drug_df = pd.DataFrame(drug_names, columns = ["DrugID"])
    drug_df["drug_index"] = drug_df.index
    # get cell ids and indices
    cell_df = pd.DataFrame(cell_ids, columns = ["CancID"])
    cell_df["cell_index"] = cell_df.index
    # create dataframe of test positions
    test_positions = pd.DataFrame(test.positions, columns = ["drug_index", "cell_index"])
    # match drug and cell id indices with test positions 
    temp_test_positions = pd.merge(test_positions, drug_df, how = "left", on = "drug_index")
    final_test_positions = pd.merge(temp_test_positions, cell_df, how = "left", on = "cell_index")
    # add normalized true values
    final_df = pd.concat([final_test_positions, pd.DataFrame(test_values, columns = ["True"])], axis=1)
    # reverse normalization of true values
    final_df["True"] = final_df["True"].apply(lambda x: math.log(((1-x)/x)**-10))
    # add normalized predicted values
    final_df = pd.concat([final_df, pd.DataFrame(test_predict, columns = ["Pred"])], axis=1)
    # reverse normalization of predicted values
    final_df["Pred"] = final_df["Pred"].apply(lambda x: math.log(((1-x)/x)**-10))
    # drop columns
    true_pred_df = final_df.drop(columns = ["drug_index", "cell_index"])
    # save predictions - long format TODO change arguments for inference
    true_pred_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

def main():
    gParameters = initialize_parameters()
    run(gParameters)
    print("Done.")

if __name__ == "__main__":
    main()
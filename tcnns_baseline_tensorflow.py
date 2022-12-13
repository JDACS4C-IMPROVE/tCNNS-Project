import os
from pathlib import Path
import candle
import tensorflow as tf
from batcher import *
import tcnns
import numpy as np
import json
from sklearn.metrics import mean_squared_error
from scipy import stats

file_path = os.path.dirname(os.path.realpath(__file__))

def weight_variable(shape, std_dev):
    initial = tf.truncated_normal(shape, stddev=std_dev)
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

# added function
def Spearman(real, predicted):
    rs = tf.py_function(stats.spearmanr, [tf.cast(predicted, tf.float32), 
                       tf.cast(real, tf.float32)], Tout = tf.float32)
    return rs

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

def run(gParameters): 

    args = candle.ArgumentStruct(**gParameters)

    if tf.test.gpu_device_name():
        print("Default GPU Device:{}".format(tf.test.gpu_device_name()))
    else:
        print("GPU not available")

    fdir = Path(__file__).resolve().parent
    if args.output_dir is not None:
        outdir = args.output_dir
    else:
        outdir = fdir/f"/results_all"
    os.makedirs(outdir, exist_ok=True)
    
    data_dir = os.getenv('CANDLE_DATA_DIR')
    #data_dir='.'
    print(data_dir)

    # get data from server or candle_
    data_file_path = candle.get_file(args.processed_data, args.data_url, datadir = data_dir, cache_subdir = None)
    
    #print(data_file_path)

    drug_smile_dict = np.load(data_dir + "/data_processed/" + args.drug_file, encoding="latin1", allow_pickle=True).item()
    drug_cell_dict = np.load(data_dir + "/data_processed/" + args.response_file, encoding="latin1", allow_pickle=True).item()
    cell_mut_dict = np.load(data_dir + "/data_processed/" + args.cell_file, encoding="latin1", allow_pickle=True).item()

    # define variables
    c_chars = drug_smile_dict["c_chars"]
    drug_names = drug_smile_dict["drug_names"]
    drug_cids = drug_smile_dict["drug_cids"]
    canonical = drug_smile_dict["canonical"]
    canonical = np.transpose(canonical, (0, 2, 1))
    cell_names = cell_mut_dict["cell_names"]
    mut_names = cell_mut_dict["mut_names"]
    cell_mut = cell_mut_dict["cell_mut"]
    all_positions = drug_cell_dict["positions"]
    np.random.shuffle(np.array(list(all_positions[()])))

    length_smiles = len(canonical[0])
    num_cell_features = len(mut_names)
    num_chars_smiles = len(c_chars)

    print("Length of SMILES string: {}".format(length_smiles)) # length of smiles
    print("Number of mutations: {}".format(num_cell_features)) # number of mutations
    print("Number of unique characters in SMILES string: {}".format(num_chars_smiles)) # number of characters in smiles

    # define model
    drug = tf.placeholder(tf.float32, shape=[None, length_smiles, num_chars_smiles])
    cell = tf.placeholder(tf.float32, shape=[None, num_cell_features])
    scores = tf.placeholder(tf.float32, shape=[None, 1])
    keep_prob = tf.placeholder(tf.float32)

    # define drug convolutional layers
    for i in range(0, len(args.drug_conv_out)):
        if i == 0:
            drug_conv_out = args.drug_conv_out[i] 
            drug_conv_pool = args.drug_pool[i]
            drug_conv_w = weight_variable([args.drug_conv_width[i], num_chars_smiles, drug_conv_out], args.std_dev)
            drug_conv_b = bias_variable([drug_conv_out], args.bias_constant)
            drug_conv_h = tf.nn.relu(conv1d(drug, drug_conv_w, args.conv_stride) + drug_conv_b)
            drug_conv_p = max_pool_1d(drug_conv_h, [drug_conv_pool], [drug_conv_pool])
        else:
            drug_conv_out = args.drug_conv_out[i] 
            drug_conv_pool = args.drug_pool[i]
            drug_conv_w = weight_variable([args.drug_conv_width[i], args.drug_conv_out[i-1], drug_conv_out], args.std_dev)
            drug_conv_b = bias_variable([drug_conv_out], args.bias_constant)
            drug_conv_h = tf.nn.relu(conv1d(drug_conv_p, drug_conv_w, args.conv_stride) + drug_conv_b)
            drug_conv_p = max_pool_1d(drug_conv_h, [drug_conv_pool], [drug_conv_pool])

    # define cell convolutional layers
    for i in range(0, len(args.cell_conv_out)):
        if i == 0:
            cell_conv_out = args.cell_conv_out[i]
            cell_conv_pool = args.cell_pool[i]
            cell_tensor = tf.expand_dims(cell, 2)
            cell_conv_w = weight_variable([args.cell_conv_width[i], 1, cell_conv_out], args.std_dev)
            cell_conv_b = weight_variable([cell_conv_out], args.bias_constant)
            cell_conv_h = tf.nn.relu(conv1d(cell_tensor, cell_conv_w, args.conv_stride) + cell_conv_b)
            cell_conv_p = max_pool_1d(cell_conv_h, [cell_conv_pool], [cell_conv_pool])
        else: 
            cell_conv_out = args.cell_conv_out[i]
            cell_conv_pool = args.cell_pool[i]
            cell_tensor = tf.expand_dims(cell, 2)
            cell_conv_w = weight_variable([args.cell_conv_width[i], args.cell_conv_out[i-1], cell_conv_out], args.std_dev)
            cell_conv_b = weight_variable([cell_conv_out], args.bias_constant)
            cell_conv_h = tf.nn.relu(conv1d(cell_conv_p, cell_conv_w, args.conv_stride) + cell_conv_b)
            cell_conv_p = max_pool_1d(cell_conv_h, [cell_conv_pool], [cell_conv_pool])

    conv_merge = tf.concat([drug_conv_p, cell_conv_p], 1)
    shape = conv_merge.get_shape().as_list()
    conv_flat = tf.reshape(conv_merge, [-1, shape[1] * shape[2]])

    # define fully connected layers
    for i in range(0, len(args.dense)):
        if i == 0:
            fc_w = weight_variable([shape[1] * shape[2], args.dense[i]], args.std_dev)
            fc_b = bias_variable([args.dense[i]], args.bias_constant)
            fc_h = tf.nn.relu(tf.matmul(conv_flat, fc_w) + fc_b)
            fc_drop = tf.nn.dropout(fc_h, keep_prob)
        elif i == (len(args.dense) - 1): 
            fc_w = weight_variable([args.dense[i], 1], args.std_dev)
            fc_b = weight_variable([1], args.std_dev)
        else:
            fc_w = weight_variable([args.dense[i], args.dense[i]], args.std_dev)
            fc_b = bias_variable([args.dense[i]], args.bias_constant)
            fc_h = tf.nn.relu(tf.matmul(fc_drop, fc_w) + fc_b)
            fc_drop = tf.nn.dropout(fc_h, keep_prob)

    y_conv = tf.nn.sigmoid(tf.matmul(fc_drop, fc_w) + fc_b)

    # define loss
    loss = tf.losses.mean_squared_error(scores, y_conv)
    # define optimizer
    train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss)

    # define metrics
    r_square = R2(scores, y_conv)
    pearson = Pearson(scores, y_conv)
    rmse = tf.sqrt(loss)
    spearman = Spearman(scores, y_conv)

    # create train, valid, and test datasets
    train, valid, test = load_data(args.batch_size, ['IC50'])

    # initialize saver object
    saver = tf.train.Saver(var_list=tf.trainable_variables())
    
    # file to store results
    output_file = open(str(outdir) + "/" + "result_all.txt", "a")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)

        test_values, test_drugs, test_cells = test.whole_batch()
        valid_values, valid_drugs, valid_cells = valid.whole_batch()
        epoch = 0
        min_loss = args.min_loss
        count = 0
        while count < 10:
            train.reset()
            step = 0
            while(train.available()):
                real_values, drug_smiles, cell_muts = train.mini_batch()
                train_step.run(feed_dict={drug:drug_smiles, cell:cell_muts, scores:real_values, keep_prob:args.dropout})
                step += 1
            valid_loss, valid_r2, valid_pcc, valid_rmse, valid_scc = sess.run([loss, r_square, pearson, rmse, spearman], feed_dict={drug:valid_drugs, cell:valid_cells, scores:valid_values, keep_prob:1})
            print("epoch: %d, loss: %g r2: %g pearson: %g rmse: %g, spearman: %g" % (epoch, valid_loss, valid_r2, valid_pcc, valid_rmse, valid_scc))
            epoch += 1
            if valid_loss < min_loss:
                test_loss, test_r2, test_pcc, test_rmse, test_scc = sess.run([loss, r_square, pearson, rmse, spearman], feed_dict={drug:test_drugs, cell:test_cells, scores:test_values, keep_prob:1})
                print("find best, loss: %g r2: %g pearson: %g rmse: %g spearman: %g******" % (test_loss, test_r2, test_pcc, test_rmse, test_scc))
                # save scores associated with lowest validation loss
                val_scores = {"val_loss": float(valid_loss), "pcc": float(valid_pcc), "scc": float(valid_scc), "rmse": float(valid_rmse)}
                os.system("rm {}/*".format(args.ckpt_directory))
                saver.save(sess, args.ckpt_directory + "/" + "result.ckpt")
                print("saved!")
                min_loss = valid_loss
                count = 0
            else:
                count = count + 1

        if test_r2 > -2:
            output_file.write("test loss: %g, test r2 %g, test pearson %g, test rmse %g, test spearman %g\n"%(test_loss, test_r2, test_pcc, test_rmse, test_scc))
            print("Saved!!!!!")
        output_file.close()
    
    # Supervisor HPO
    print("\nIMPROVE_RESULT val_loss:\t{}\n".format(val_scores["val_loss"]))
    with open(Path(args.output_dir) / "scores.json", "w", encoding="utf-8") as f:
        json.dump(val_scores, f, ensure_ascii=False, indent=4)

def main():
    gParameters = initialize_parameters()
    run(gParameters)
    print("Done.")

if __name__ == "__main__":
    main()

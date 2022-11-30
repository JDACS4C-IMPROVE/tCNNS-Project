import os
from pathlib import Path
import candle
import tensorflow as tf
from batcher import *
import tcnns
import numpy as np

file_path = os.path.dirname(os.path.realpath(__file__))

def weight_variable(shape, std_dev):
    initial = tf.truncated_normal(shape, stddev=std_dev) # added
    return tf.Variable(initial)

def bias_variable(shape, bias_constant):
    initial = tf.constant(bias_constant, shape=shape) # added
    return tf.Variable(initial) 

def conv1d(x, W, conv_stride):
    return tf.nn.conv1d(x, W, conv_stride, padding='SAME') # added

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

def run(args): 

    if tf.test.gpu_device_name():
        print("Default GPU Device:{}".format(tf.test.gpu_device_name()))
    else:
        print("GPU not available")

    fdir = Path(__file__).resolve().parent
    if args.output_dir is not None:
        outdir = fdir/args.output_dir
    else:
        outdir = fdir/f"/results_all"
    os.makedirs(outdir, exist_ok=True)

    """
    # fetch data (if needed)
    ftp_origin = args.data_url
    data_file_list = ["drug_onehot_smiles.npy", "drug_cell_interaction.npy", "cell_mut_matrix.npy"]
    for f in data_file_list:
        candle.get_file(fname=f.strip(),
                        origin=os.path.join(ftp_origin, f.strip()),
                        unpack=False, md5_hash=None,
                        datadir=fdir/f"./data_processed",
                        cache_subdir=None)
    """

    # load files
    drug_smile_dict = np.load(str(args.data_dir) + "/" + "drug_onehot_smiles.npy", encoding="latin1", allow_pickle=True).item()
    drug_cell_dict = np.load(str(args.data_dir) + "/" + "drug_cell_interaction.npy", encoding="latin1", allow_pickle=True).item()
    cell_mut_dict = np.load(str(args.data_dir) + "/" + "cell_mut_matrix.npy", encoding="latin1", allow_pickle=True).item()

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
    np.random.shuffle(all_positions)

    # define model
    drug = tf.placeholder(tf.float32, shape=[None, args.length_smiles, args.num_chars_smiles]) # added
    cell = tf.placeholder(tf.float32, shape=[None, args.num_cell_features]) # added
    scores = tf.placeholder(tf.float32, shape=[None, 1])
    keep_prob = tf.placeholder(tf.float32)

    drug_conv1_out = args.drug_conv[0] # added
    drug_conv1_pool = args.pool[0] # added
    drug_conv1_w = weight_variable([args.conv_width[0], args.num_chars_smiles, drug_conv1_out], args.std_dev) # added 7 and 28
    drug_conv1_b = bias_variable([drug_conv1_out], args.bias_constant)
    drug_conv1_h = tf.nn.relu(conv1d(drug, drug_conv1_w, args.conv_stride) + drug_conv1_b)
    drug_conv1_p = max_pool_1d(drug_conv1_h, [drug_conv1_pool], [drug_conv1_pool])

    drug_conv2_out = args.drug_conv[1] # added
    drug_conv2_pool = args.pool[1] # added
    drug_conv2_w = weight_variable([args.conv_width[1], drug_conv1_out, drug_conv2_out], args.std_dev) # added 7
    drug_conv2_b = bias_variable([drug_conv2_out], args.bias_constant)
    drug_conv2_h = tf.nn.relu(conv1d(drug_conv1_p, drug_conv2_w, args.conv_stride) + drug_conv2_b)
    drug_conv2_p = max_pool_1d(drug_conv2_h, [drug_conv2_pool], [drug_conv2_pool])

    drug_conv3_out = args.drug_conv[2] # added
    drug_conv3_pool = args.pool[2] # added
    drug_conv3_w = weight_variable([args.conv_width[2], drug_conv2_out, drug_conv3_out], args.std_dev) # added 7
    drug_conv3_b = bias_variable([drug_conv3_out], args.bias_constant)
    drug_conv3_h = tf.nn.relu(conv1d(drug_conv2_p, drug_conv3_w, args.conv_stride) + drug_conv3_b)
    drug_conv3_p = max_pool_1d(drug_conv3_h, [drug_conv3_pool], [drug_conv3_pool])

    cell_conv1_out = args.cell_conv[0] # added
    cell_conv1_pool = args.pool[3] # added
    cell_tensor = tf.expand_dims(cell, 2)
    cell_conv1_w = weight_variable([args.conv_width[3], 1, cell_conv1_out], args.std_dev) # added 7
    cell_conv1_b = weight_variable([cell_conv1_out], args.bias_constant)
    cell_conv1_h = tf.nn.relu(conv1d(cell_tensor, cell_conv1_w, args.conv_stride) + cell_conv1_b)
    cell_conv1_p = max_pool_1d(cell_conv1_h, [cell_conv1_pool], [cell_conv1_pool])

    cell_conv2_out = args.cell_conv[1] # added
    cell_conv2_pool = args.pool[4] # added
    cell_conv2_w = weight_variable([args.conv_width[4], cell_conv1_out, cell_conv2_out], args.std_dev) # added 7
    cell_conv2_b = bias_variable([cell_conv2_out], args.bias_constant)
    cell_conv2_h = tf.nn.relu(conv1d(cell_conv1_p, cell_conv2_w, args.conv_stride) + cell_conv2_b)
    cell_conv2_p = max_pool_1d(cell_conv2_h, [cell_conv2_pool], [cell_conv2_pool])

    cell_conv3_out = args.cell_conv[2] # added
    cell_conv3_pool = args.pool[5] # added
    cell_conv3_w = weight_variable([args.conv_width[5], cell_conv2_out, cell_conv3_out], args.std_dev) # added 7
    cell_conv3_b = bias_variable([cell_conv3_out], args.bias_constant)
    cell_conv3_h = tf.nn.relu(conv1d(cell_conv2_p, cell_conv3_w, args.conv_stride) + cell_conv3_b)
    cell_conv3_p = max_pool_1d(cell_conv3_h, [cell_conv3_pool], [cell_conv3_pool])

    conv_merge = tf.concat([drug_conv3_p, cell_conv3_p], 1)
    shape = conv_merge.get_shape().as_list()
    conv_flat = tf.reshape(conv_merge, [-1, shape[1] * shape[2]])

    fc1_w = weight_variable([shape[1] * shape[2], args.dense[0]]) # added
    fc1_b = bias_variable(args.dense[0]) # added
    fc1_h = tf.nn.relu(tf.matmul(conv_flat, fc1_w) + fc1_b)
    fc1_drop = tf.nn.dropout(fc1_h, keep_prob)

    fc2_w = weight_variable([args.dense[1], args.dense[1]]) # added
    fc2_b = bias_variable([args.dense[1]]) # added
    fc2_h = tf.nn.relu(tf.matmul(fc1_drop, fc2_w) + fc2_b)
    fc2_drop = tf.nn.dropout(fc2_h, keep_prob)

    fc3_w = weight_variable([args.dense[2], 1]) # added
    fc3_b = weight_variable([1])

    y_conv = tf.nn.sigmoid(tf.matmul(fc2_drop, fc3_w) + fc3_b)

    # define loss
    loss = tf.losses.mean_squared_error(scores, y_conv)
    # define optimizer
    train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(loss) # added learning rate

    # define metrics
    r_square = R2(scores, y_conv)
    pearson = Pearson(scores, y_conv)
    rmsr = tf.sqrt(loss)

    # create train, valid, and test datasets
    train, valid, test = load_data(args.batch_size, ['IC50']) # added batch size

    # initialize saver object
    saver = tf.train.Saver(var_list=tf.trainable_variables())
    
    # file to store results
    output_file = open(str(outdir) + "/" + "result_all.txt", "a") # added output_dir

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_values, test_drugs, test_cells = test.whole_batch()
        valid_values, valid_drugs, valid_cells = valid.whole_batch()
        epoch = 0
        min_loss = args.min_loss # added
        count = 0
        while count < 10:
            train.reset()
            step = 0
            while(train.available()):
                real_values, drug_smiles, cell_muts = train.mini_batch()
                train_step.run(feed_dict={drug:drug_smiles, cell:cell_muts, scores:real_values, keep_prob:args.dropout}) # added
                step += 1
            valid_loss, valid_r2, valid_p, valid_rmsr = sess.run([loss, r_square, pearson, rmsr], feed_dict={drug:valid_drugs, cell:valid_cells, scores:valid_values, keep_prob:1})
            print("epoch: %d, loss: %g r2: %g pearson: %g rmsr: %g" % (epoch, valid_loss, valid_r2, valid_p, valid_rmsr))
            epoch += 1
            if valid_loss < min_loss:
                test_loss, test_r2, test_p, test_rmsr = sess.run([loss, r_square, pearson, rmsr], feed_dict={drug:test_drugs, cell:test_cells, scores:test_values, keep_prob:1})
                print("find best, loss: %g r2: %g pearson: %g rmsr: %g ******" % (test_loss, test_r2, test_p, test_rmsr))
                os.system("rm {}/*".format(args.ckpt.directory)) # added checkpoint directory
                saver.save(sess, args.ckpt.directory/f"result.ckpt")
                print("saved!")
                min_loss = valid_loss
                count = 0
            else:
                count = count + 1

        if test_r2 > -2:
            output_file.write("%g,%g,%g,%g\n"%(test_loss, test_r2, test_p, test_rmsr))
            print("Saved!!!!!")
        output_file.close()



def main():
    gParameters = initialize_parameters()
    args = candle.ArgumentStruct(**gParameters)
    print(args)
    run(args)

if __name__ == "__main__":
    main()


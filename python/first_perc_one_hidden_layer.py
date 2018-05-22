from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

# additional imports
import os
import time
import scipy.misc
import matplotlib.pyplot as plt

savedir = 'ml_output_tensorflow'

# -------------------------------------------------------------------------------
# functions

# save loss history


def save_loss_history(savedir, name, data):
    plt.figure(name)
    plt.plot(range(len(data)), data)
    # plt.show()
    plt.savefig(savedir + '/' + name + '.png')
    plt.gcf().clear()


def create_dir(savedir):
    try:
        os.mkdir(savedir)
    except:
        print(savedir, "already created")


def show_loss_history(savedir, name, data):
    plt.figure(name)
    plt.plot(range(len(data)), data)
    plt.show()
    #plt.savefig(savedir + '/' + name + '.png')
    # plt.gcf().clear()


def print_dist(savedir, name, plotdata):
    plt.figure(name)
    plt.title(name)
    plt.hist(plotdata, normed=True, bins=30)
#    plt.show()
    plt.savefig(savedir + '/' + name + '.png')
    plt.gcf().clear()

# -------------------------------------------------------------------------------
# data loader


#### ---- Marker 1 ---- ####
# get pickle file
pickle_file = '/Users/christoph/Documents/coding/CERN_input_data/python/data_root/5para.pickle'

# open pickle file and datasets & targets
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_targets = save['train_targets']
    valid_dataset = save['valid_dataset']
    valid_targets = save['valid_targets']
    test_dataset = save['test_dataset']
    test_targets = save['test_targets']
    del save  # hint to help gc free up memory

    print('Training set', train_dataset.shape, train_targets.shape)
    print('Validation set', valid_dataset.shape, valid_targets.shape)
    print('Test set', test_dataset.shape, test_targets.shape)

    train_dataset = train_dataset.astype(np.float32, copy=False)
    train_targets = train_targets.astype(np.float32, copy=False)
    valid_dataset = valid_dataset.astype(np.float32, copy=False)
    valid_targets = valid_targets.astype(np.float32, copy=False)
    test_dataset = test_dataset.astype(np.float32, copy=False)
    test_targets = test_targets.astype(np.float32, copy=False)

    print('train_dataset type', train_dataset.dtype)
    print('train_targets type', train_targets.dtype)
    print('valid_dataset type', valid_dataset.dtype)
    print('valid_targets type', valid_targets.dtype)
    print('test_dataset type', test_dataset.dtype)
    print('test_targets type', test_targets.dtype)


#### ---- Marker 2 ---- ####
# timer
start = time.clock()
currenttime = time.clock()
print('\nstart time: ' + str(start))


# -------------------------------------------------------------------------------
# load all the data into TensorFlow and build the computation graph corresponding to our training

train_dataset_TMP = train_dataset
train_targets_TMP = train_targets
valid_dataset_TMP = valid_dataset
valid_targets_TMP = valid_targets
test_dataset_TMP = test_dataset
test_targets_TMP = test_targets


#### ---- Marker 3 ---- ####
for test_iterations in range(1, 10):
    # reduce size of data set
    #### ---- Marker 4 ---- ####
    print("!!!! test_iterations", test_iterations, "!!!!")
    train_reduced_size = 10000*test_iterations
    valid_reduced_size = int(train_reduced_size*0.1)
    test_reduced_size = int(train_reduced_size*0.1)

    train_dataset = train_dataset_TMP[0:train_reduced_size, :]
    train_targets = train_targets_TMP[0:train_reduced_size, :]
    valid_dataset = valid_dataset_TMP[0:valid_reduced_size, :]
    valid_targets = valid_targets_TMP[0:valid_reduced_size, :]
    test_dataset = test_dataset_TMP[0:test_reduced_size, :]
    test_targets = test_targets_TMP[0:test_reduced_size, :]

    #### ---- Marker 5 ---- ####
    # Hyperparameters
    # num_steps = 3001
    num_steps = 100001
    learning_rate = 0.1  # standard: 0.1
    momentum = 0.9  # standard: 0.9

    para_dataset_size = 72
    para_targets_size = 5
    batch_size = 1024
    num_nodes = 128

    #### ---- Marker 6 ---- ####
    # additional tensors
    loss_history = np.empty(shape=[1], dtype=float)
    final_logits = np.empty(shape=train_targets.shape, dtype=float)

    #### ---- Marker 7 ---- ####
    graph = tf.Graph()
    with graph.as_default():

        #### ---- Marker 8 ---- ####
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, para_dataset_size))
        tf_train_targets = tf.placeholder(
            tf.float32, shape=(batch_size, para_targets_size))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        weights_1 = tf.Variable(
            tf.truncated_normal([para_dataset_size, num_nodes]))
        biases_1 = tf.Variable(tf.zeros([num_nodes]))
        weights_2 = tf.Variable(tf.truncated_normal(
            [num_nodes, para_targets_size]))
        biases_2 = tf.Variable(tf.zeros([para_targets_size]))

        #### ---- Marker 9 ---- ####
        # Training computation.
        logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1

        # Hidden layer
        # Activation Function
        act_layer = tf.nn.tanh(logits_1)

        logits_2 = tf.matmul(act_layer, weights_2) + biases_2

        print('tf_train_dataset', tf_train_dataset.shape)
        print('tf_train_targets', tf_train_targets.shape)
        print('weights_1', weights_1.shape)
        print('biases_1', biases_1.shape)
        print('logits_1', logits_1.shape)
        print('act_layer', act_layer.shape)
        print('weights_2', weights_2.shape)
        print('biases_2', biases_2.shape)
        print('logits_2', logits_2.shape)

        # Quadratic Loss Function
        loss = tf.reduce_mean(tf.square(tf_train_targets - logits_2))
        print('loss', loss.shape)

        #### ---- Marker 10 ---- ####
        # Optimizer & Training Step
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        optimizer = tf.train.MomentumOptimizer(
            learning_rate, momentum, use_locking=False, use_nesterov=False).minimize(loss)
    #    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum,use_locking=False,use_nesterov=True).minimize(loss)

    # -------------------------------------------------------------------------------
    # run computation and iterate

    #### ---- Marker 11 ---- ####
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print("Initialized")
        #### ---- Marker 12 ---- ####
        for step in range(num_steps):

            #### ---- Marker 13 ---- ####
            offset = np.random.randint(0, train_targets.shape[0] - batch_size)

            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_targets = train_targets[offset:(offset + batch_size), :]

            #### ---- Marker 14 ---- ####
            # Prepare a dictionary telling the session where to feed the minibatch.
            feed_dict = {tf_train_dataset: batch_data,
                         tf_train_targets: batch_targets}

            #### ---- Marker 15 ---- ####
            _, l = session.run(
                [optimizer, loss], feed_dict=feed_dict)

            # Prints to keep track of progress
            if (step % 500 == 0):
                print("\nMinibatch loss at step %d: %f" % (step, l))
                print('Computing time for steps ' + str(step-500) + ' to ' +
                      str(step) + ': ' + str(time.clock()-currenttime))
                currenttime = time.clock()
                print("offset", offset)

            #### ---- Marker 16 ---- ####
            # Append current loss to loss history
            loss_history = np.append(loss_history, l)

            tf_train_dataset_out, weights_1_out, biases_1_out, logits_1_out, act_layer_out, weights_2_out, biases_2_out, logits_2_out, tf_train_targets_out = session.run(
                [tf_train_dataset, weights_1, biases_1, logits_1, act_layer, weights_2, biases_2, logits_2, tf_train_targets], feed_dict=feed_dict)

            #### ---- Marker 17 ---- ####
            # Update final_logits for difference calculator
            final_logits[offset:(offset + batch_size), :] = logits_2_out

        print("\nfinal loss: %f" % l)
        print('total computing time: ' + str(time.clock()-start))

    #    saver = tf.train.Saver()
    #    saver.save(session, 'session_store/a2_sgd.ckpt')

    # -------------------------------------------------------------------------------
    # analysis

    now = time.localtime()
    time_now = str(now.tm_year) + "." + str(now.tm_mon) + "." + str(now.tm_mday) + \
        "_" + str(now.tm_hour) + "." + str(now.tm_min) + "." + str(now.tm_sec)

    savedir_with_time = savedir + "/" + time_now

    #### ---- Marker 18 ---- ####
    # create folder
    create_dir(savedir_with_time)

    #### ---- Marker 19 ---- ####
    # loss history prints
    if (num_steps < 101):
        save_loss_history(savedir_with_time, 'loss_history', loss_history)
    if (num_steps > 2001):
        save_loss_history(savedir_with_time, 'loss_history_1001_to_end_steps',
                          loss_history[1001:len(loss_history)])
    if (num_steps > 10001):
        save_loss_history(savedir_with_time, 'loss_history_5001_to_end_steps',
                          loss_history[5001:len(loss_history)])
    if (num_steps > 20001):
        save_loss_history(savedir_with_time, 'loss_history_10001_to_end_steps',
                          loss_history[10001:len(loss_history)])

    save_loss_history(savedir_with_time, 'loss_history_first_1000_steps',
                      loss_history[0:1000])
    save_loss_history(savedir_with_time, 'loss_history_1001_to_end_steps',
                      loss_history[1001:len(loss_history)])
    save_loss_history(savedir_with_time, 'loss_history_last_1000_steps',
                      loss_history[len(loss_history)-1000:len(loss_history)])

    #### ---- Marker 20 ---- ####
    # difference calculator
    try:
        diff = final_logits-train_targets
        diff_mean = np.mean(np.transpose(diff), 1)
        diff_std = np.std(np.transpose(diff), 1)
        print("diff_std", diff_std)
    except:
        print("diff didn't work")

    # save mean & std
    try:
        f = open(savedir_with_time + "/tf_output_benchmarks.csv", "w")
        for i in range(0, diff_mean.shape[0]):
            f.write(str(diff_mean[i]))
            f.write(', ')
            f.write(str(diff_std[i]))
            f.write('\n')
        f.close()
    except:
        print("benchmark 1 didn't work")

    try:
        f = open(savedir + "/tf_output_benchmarks.csv", "w")
        for i in range(0, diff_mean.shape[0]):
            f.write(str(diff_mean[i]))
            f.write(', ')
            f.write(str(diff_std[i]))
            f.write('\n')
        f.close()
    except:
        print("benchmark 2 didn't work")

    # print histos
    try:
        labels = ['q_p', 'dx_dz', 'dy_dz', 'x', 'y']
        for i in range(0, 5):
            name = 'Tensorflow_ML_Output_Para_' + str(i+1) + "_" + labels[i]
            print_dist(savedir_with_time, name, diff[:, i])
    except:
        print("histos didn't work")

    #### ---- Marker 21 ---- ####
    # write to log
    try:
        f = open(savedir + "/tf_output_log.csv", "a")
        f.write(str(diff_std[2]))
        f.write(', ')
        f.write(str(train_dataset.shape[0]))
        f.write(', ')
        f.write(str(valid_dataset.shape[0]))
        f.write(', ')
        f.write(str(test_dataset.shape[0]))
        f.write(', ')
        f.write(str(para_dataset_size))
        f.write(', ')
        f.write(str(para_targets_size))
        f.write(', ')
        f.write(str(num_steps))
        f.write(', ')
        f.write(str(learning_rate))
        f.write(', ')
        f.write(str(momentum))
        f.write(', ')
        f.write(str(batch_size))
        f.write(', ')
        f.write(str(num_nodes))
        f.write(', ')
        for i in range(0, diff_mean.shape[0]):
            f.write(str(diff_mean[i]))
            f.write(', ')
        f.write(' // ')
        for i in range(0, diff_mean.shape[0]):
            f.write(str(diff_std[i]))
            f.write(', ')
        f.write('\n')
        f.close()
    except:
        print("log didn't work")

print('programm terminated.')

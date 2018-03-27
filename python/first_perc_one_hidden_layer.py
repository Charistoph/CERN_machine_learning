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

#-------------------------------------------------------------------------------
# functions

# save loss history
def save_loss_history(savedir,name,data):
    plt.figure(name)
    plt.plot(range(len(data)),data)
    #plt.show()
    plt.savefig(savedir + '/' + name + '.png')
    plt.gcf().clear()

# show loss history
def show_loss_history(savedir,name,data):
    plt.figure(name)
    plt.plot(range(len(data)),data)
    plt.show()
    #plt.savefig(savedir + '/' + name + '.png')
    #plt.gcf().clear()

def print_dist(savedir,name,plotdata):
    plt.figure(name)
    plt.title(name)
    plt.hist(plotdata, normed=True, bins=30)
#    plt.show()
    plt.savefig(savedir + '/' + name + '.png')
    plt.gcf().clear()

#-------------------------------------------------------------------------------
# data loader

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

# timer
start = time.clock()
currenttime = time.clock()
print ('\nstart time: ' + str(start))


#-------------------------------------------------------------------------------
# load all the data into TensorFlow and build the computation graph corresponding to our training

# settings
para_dataset_size = 72
para_targets_size = 5
batch_size = 1024
num_nodes = 128
learning_rate = 0.1
momentum = 0.5

# additional tensors
loss_history = np.empty(shape=[1],dtype=float)
final_logits = np.empty(shape=train_targets.shape,dtype=float)

graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, para_dataset_size))
    tf_train_targets = tf.placeholder(tf.float32, shape=(batch_size, para_targets_size))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights_1 = tf.Variable(
        tf.truncated_normal([para_dataset_size, num_nodes]))
    biases_1 = tf.Variable(tf.zeros([num_nodes]))
    weights_2 = tf.Variable(tf.truncated_normal([num_nodes, para_targets_size]))
    biases_2 = tf.Variable(tf.zeros([para_targets_size]))

    # Training computation.
    logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1

    # Hidden layer
    # RELU
    relu_layer = tf.nn.tanh(logits_1)

    logits_2 = tf.matmul(relu_layer, weights_2) + biases_2

    print('tf_train_dataset', tf_train_dataset.shape)
    print('tf_train_targets', tf_train_targets.shape)
    print('weights_1', weights_1.shape)
    print('biases_1', biases_1.shape)
    print('logits_1', logits_1.shape)
    print('relu_layer', relu_layer.shape)
    print('weights_2', weights_2.shape)
    print('biases_2', biases_2.shape)
    print('logits_2', logits_2.shape)

    # Quadratic Loss Function
    loss = tf.reduce_mean(tf.square(tf_train_targets - logits_2))
    print('loss', loss.shape)

    # Optimizer & Training Step
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum,use_locking=False,use_nesterov=False).minimize(loss)
#    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum,use_locking=False,use_nesterov=True).minimize(loss)


#-------------------------------------------------------------------------------
# run computation and iterate
#num_steps = 3001
num_steps = 3001

def accuracy(predictions, targets):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(targets, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_targets.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_targets = train_targets[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_targets : batch_targets}
        _, l = session.run(
          [optimizer, loss], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("\nMinibatch loss at step %d: %f" % (step, l))
            print('Computing time for steps ' + str(step-500) + ' to ' +
                    str(step) + ': ' + str(time.clock()-currenttime))
            currenttime = time.clock()

        # Append current loss to loss history
        loss_history = np.append(loss_history,l)

        tf_train_dataset_out,weights_1_out,biases_1_out,logits_1_out,relu_layer_out,weights_2_out,biases_2_out,logits_2_out,tf_train_targets_out = session.run([tf_train_dataset,weights_1,biases_1,logits_1,relu_layer,weights_2,biases_2,logits_2,tf_train_targets], feed_dict=feed_dict)

        # Update final_logits for difference calculator
        final_logits[offset:(offset + batch_size), :] = logits_2_out

#        if (step < 3 or step > num_steps-2):
#            print('\nStep:', step)
#            print('tf_train_dataset_out\n', tf_train_dataset_out[0:5,0])
#            print('weights_1_out\n', weights_1_out[0:5,0])
#            print('biases_1_out\n', biases_1_out[0:5])
#            print('logits_1_out\n', logits_1_out[0:5,0])
#            print('relu_layer_out\n', relu_layer_out[0:5,0])
#            print('weights_2_out\n', weights_2_out[0:5,0])
#            print('biases_2_out\n', biases_2_out[0:5])
#            print('logits_2_out\n', logits_2_out[0:5,0])
#            print('tf_train_targets_out\n', tf_train_targets_out[0:5,0])
#            print('loss',l)
#        '''
    print("\nfinal loss: %f" % l)
    print ('total computing time: ' + str(time.clock()-start))

#    saver = tf.train.Saver()
#    saver.save(session, 'session_store/a2_sgd.ckpt')

#-------------------------------------------------------------------------------
# analysis

# loss history prints
if (num_steps<101):
    save_loss_history(savedir,'loss_history',loss_history)
if (num_steps>2001):
    save_loss_history(savedir,'loss_history_1001_to_end_steps',loss_history[1001:len(loss_history)])
if (num_steps>10001):
    save_loss_history(savedir,'loss_history_5001_to_end_steps',loss_history[5001:len(loss_history)])
if (num_steps>20001):
    save_loss_history(savedir,'loss_history_10001_to_end_steps',loss_history[10001:len(loss_history)])
else:
    save_loss_history(savedir,'loss_history_first_100_steps',loss_history[0:100])
    save_loss_history(savedir,'loss_history_101_to_end_steps',loss_history[101:len(loss_history)])
    save_loss_history(savedir,'loss_history_last_100_steps',loss_history[len(loss_history)-100:len(loss_history)])

# difference calculator
diff=final_logits-train_targets
diff_mean = np.mean(np.transpose(diff),1)
diff_std = np.std(np.transpose(diff),1)

# save mean & std
f = open("ml_output_tensorflow/tf_output_benchmarks.csv","w")
for i in range(0,diff_mean.shape[0]):
    f.write(str(diff_mean[i]))
    f.write(', ')
    f.write(str(diff_std[i]))
    f.write('\n')
f.close()

# print histos
labels = ['q_p','dx_dz','dy_dz','x','y']
for i in range(0,5):
    name = 'Tensorflow_ML_Output_Para_' + str(i+1) + "_" + labels[i]
    print_dist(savedir,name,diff[:,i])

print('programm terminated.')

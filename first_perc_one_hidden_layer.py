from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

# additional imports
import os
import time
import scipy.misc

# get pickle file
pickle_file = 'data_root/5para.pickle'

# open pickle file and datasets & labels
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    print('train_dataset type', train_dataset.dtype)
    print('train_labels type', train_labels.dtype)
    print('valid_dataset type', valid_dataset.dtype)
    print('valid_labels type', valid_labels.dtype)
    print('test_dataset type', test_dataset.dtype)
    print('test_labels type', test_labels.dtype)

    train_dataset = train_dataset.astype(np.float32, copy=False)
    train_labels = train_labels.astype(np.float32, copy=False)
    valid_dataset = valid_dataset.astype(np.float32, copy=False)
    valid_labels = valid_labels.astype(np.float32, copy=False)
    test_dataset = test_dataset.astype(np.float32, copy=False)
    test_labels = test_labels.astype(np.float32, copy=False)

    print('train_dataset type', train_dataset.dtype)
    print('train_labels type', train_labels.dtype)
    print('valid_dataset type', valid_dataset.dtype)
    print('valid_labels type', valid_labels.dtype)
    print('test_dataset type', test_dataset.dtype)
    print('test_labels type', test_labels.dtype)

# timer
start = time.clock()
currenttime = time.clock()
print ('\nstart time: ' + str(start))


#-------------------------------------------------------------------------------
# load all the data into TensorFlow and build the computation graph corresponding to our training
para_dataset_size = 72
para_labels_size = 5
batch_size = 128
num_nodes= 1024
learning_rate = 0.5

graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, para_dataset_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, para_labels_size))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights_1 = tf.Variable(
        tf.truncated_normal([para_dataset_size, num_nodes]))
    biases_1 = tf.Variable(tf.zeros([num_nodes]))
    weights_2 = tf.Variable(tf.truncated_normal([num_nodes, para_labels_size]))
    biases_2 = tf.Variable(tf.zeros([para_labels_size]))

    # Training computation.
    logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1

    # Hidden layer
    # RELU
    relu_layer = tf.nn.relu(logits_1)

    logits_2 = tf.matmul(relu_layer, weights_2) + biases_2

    # Quadratic Loss Function
#    loss = tf.reduce_mean(
#        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_2))

    loss = tf.reduce_mean(tf.square(tf_train_labels - logits_2))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
#    train_prediction = tf.nn.softmax(logits_1)
#    valid_prediction = tf.nn.softmax(
#        tf.matmul(tf_valid_dataset, weights_1) + biases_1)
#    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights_1) + biases_1)

    # Predictions for the training
    train_prediction = tf.nn.softmax(logits_2)

    # Predictions for validation
    logits_1 = tf.matmul(tf_valid_dataset, weights_1) + biases_1
    relu_layer= tf.nn.relu(logits_1)
    logits_2 = tf.matmul(relu_layer, weights_2) + biases_2

    valid_prediction = tf.nn.softmax(logits_2)

    # Predictions for test
    logits_1 = tf.matmul(tf_test_dataset, weights_1) + biases_1
    relu_layer= tf.nn.relu(logits_1)
    logits_2 = tf.matmul(relu_layer, weights_2) + biases_2

    test_prediction =  tf.nn.softmax(logits_2)

#-------------------------------------------------------------------------------
# run computation and iterate
#num_steps = 3001
num_steps = 3001

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
            print('Computing time for steps ' + str(step-500) + ' to ' +
                    str(step) + ': ' + str(time.clock()-currenttime))
            currenttime = time.clock()

    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
    print ('total computing time: ' + str(time.clock()-start))

#    saver = tf.train.Saver()
#    saver.save(session, 'session_store/a2_sgd.ckpt')

print('programm terminated.')

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

# get pickle file
pickle_file = 'data_root/5para.pickle'

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

    print('train_dataset type', train_dataset.dtype)
    print('train_targets type', train_targets.dtype)
    print('valid_dataset type', valid_dataset.dtype)
    print('valid_targets type', valid_targets.dtype)
    print('test_dataset type', test_dataset.dtype)
    print('test_targets type', test_targets.dtype)

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
para_dataset_size = 72
para_targets_size = 5
batch_size = 128
num_nodes= 1024
learning_rate = 0.1

loss_history = np.empty(shape=[1],dtype=float)

graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, para_dataset_size))
    tf_train_targets = tf.placeholder(tf.float32, shape=(batch_size, para_targets_size))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

#-------------------------------------------------------------------------------
#    '''
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
#    loss = tf.reduce_mean(
#        tf.nn.softmax_cross_entropy_with_logits(targets=tf_train_targets, logits=logits_1))


#    print('tf_train_targets', tf_train_targets)
#    print('logits_2', logits_2)
#    t1 = tf_train_targets - logits_2
#    print('t1.eval', t1.eval)
#    p1 = tf.Print(t1, message="t1")

    loss = tf.reduce_mean(tf.square(tf_train_targets - logits_2))
    print('loss', loss.shape)

#    print('loss.eval', loss.eval)
#    p2 = tf.Print(loss, message="loss")

#    model = tf.global_variables_initializer()
#    with tf.Session() as session:
#        session.run(model)
#        #print(session.run(loss))
#        print(session.run(tf_train_targets - logits_2))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits_1)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights_1) + biases_1)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights_1) + biases_1)

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
    '''
#-------------------------------------------------------------------------------

    # Variables.
    weights = tf.Variable(
        tf.truncated_normal([para_dataset_size, para_targets_size]))
    biases = tf.Variable(tf.zeros([para_targets_size]))

    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases

#    loss = tf.reduce_mean(
#        tf.nn.softmax_cross_entropy_with_logits(targets=tf_train_targets, ogits=logits))
    loss = tf.reduce_mean(tf.square(tf_train_targets - logits))

    print('tf_train_dataset', tf_train_dataset.shape)
    print('tf_train_targets', tf_train_targets.shape)
    print('weights', weights.shape)
    print('biases', biases.shape)
    print('logits', logits.shape)
    print('loss', loss.shape)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
    '''
#-------------------------------------------------------------------------------
# run computation and iterate
#num_steps = 3001
num_steps = 300

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
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("\nMinibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_targets))
            print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_targets))
            print('Computing time for steps ' + str(step-500) + ' to ' +
                    str(step) + ': ' + str(time.clock()-currenttime))
            currenttime = time.clock()

        # Append current loss to loss history
        loss_history = np.append(loss_history,l)
        '''
        tf_train_dataset_v,weights_1_v,biases_1_v,logits_1_v,tf_train_targets_v = session.run([tf_train_dataset,weights,biases,logits,tf_train_targets], feed_dict=feed_dict)

        if (step < 3 or step > num_steps-2):
            print('\nStep:', step)
            print("Minibatch loss at step %d: %f" % (step, l))
            print('tf_train_dataset_v', tf_train_dataset_v[0:5,0])
            print('weights_1_v', weights_1_v[0:5,0])
            print('biases_1_v', biases_1_v[0:5])
            print('logits_1_v', logits_1_v[0:5,0])
            print('tf_train_targets_v', tf_train_targets_v[0:5,0])
            print('loss',l)
        '''
        tf_train_dataset_v,weights_1_v,biases_1_v,logits_1_v,relu_layer_v,weights_2_v,biases_2_v,logits_2_v,tf_train_targets_v = session.run([tf_train_dataset,weights_1,biases_1,logits_1,relu_layer,weights_2,biases_2,logits_2,tf_train_targets], feed_dict=feed_dict)

        if (step < 3 or step > num_steps-2):
            print('\nStep:', step)
            print('tf_train_dataset_v\n', tf_train_dataset_v[0:5,0])
            print('weights_1_v\n', weights_1_v[0:5,0])
            print('biases_1_v\n', biases_1_v[0:5])
            print('logits_1_v\n', logits_1_v[0:5,0])
            print('relu_layer_v\n', relu_layer_v[0:5,0])
            print('weights_2_v\n', weights_2_v[0:5,0])
            print('biases_2_v\n', biases_2_v[0:5])
            print('logits_2_v\n', logits_2_v[0:5,0])
            print('tf_train_targets_v\n', tf_train_targets_v[0:5,0])
            print('loss\n',l)
#        '''
    print("\nTest accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_targets))
    print ('total computing time: ' + str(time.clock()-start))


#    saver = tf.train.Saver()
#    saver.save(session, 'session_store/a2_sgd.ckpt')

plt.figure('loss_history')
plt.plot(range(len(loss_history)),loss_history)
#plt.show()
plt.savefig('ml_output/loss_history.png')
plt.gcf().clear()

print('programm terminated.')

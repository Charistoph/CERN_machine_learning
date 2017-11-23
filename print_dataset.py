import os
import time
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle

savedir ="distribution_histo_plots"

#-------------------------------------------------------------------------------
# functions

def get_data():
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

#    print('Training set', train_dataset.shape, train_labels.shape)
#    print('Validation set', valid_dataset.shape, valid_labels.shape)
#    print('Test set', test_dataset.shape, test_labels.shape)

    total_dataset = train_dataset.shape[0] + valid_dataset.shape[0] + test_dataset.shape[0]
    inputheigth = train_dataset.shape[1]
    labelheigth = train_labels.shape[1]

    train_size = train_dataset.shape[0]
    valid_size = valid_dataset.shape[0]
    test_size = test_dataset.shape[0]

#    print('total dataset length', total_dataset)
#    print('inputheigth', inputheigth)
#    print('labelheigth', labelheigth)
#    print('train_size', train_size)
#    print('valid_size', valid_size)
#    print('test_size', test_size)

    inputs = np.zeros(shape=(total_dataset,inputheigth))
    inputs_total = np.zeros(shape=(total_dataset,5))
    labels = np.zeros(shape=(total_dataset,labelheigth))

#    print('inputs.shape', inputs.shape)
#    print('labels.shape', labels.shape)

# write split datasets into single dataset, save for labels
    inputs[0:train_size,:] = train_dataset
    labels[0:train_size,:] = train_labels
    inputs[train_size:train_size+valid_size,:] = valid_dataset
    labels[train_size:train_size+valid_size,:] = valid_labels
    inputs[train_size+valid_size:valid_size+train_size+test_size,:] = test_dataset
    labels[train_size+valid_size:valid_size+train_size+test_size,:] = test_labels

    return inputs,inputs_total,labels

def create_dir(savedir):
    try:
        os.mkdir(savedir)
    except:
        print(savedir, "already created")

def print_dist(savedir,name,plotdata):
    plt.figure(name)
    plt.hist(plotdata, normed=True, bins=30)
#    plt.show()
    plt.savefig(savedir + '/' + name + '.png')
    plt.gcf().clear()

#-------------------------------------------------------------------------------
# main code

inputs,inputs_total,labels = get_data()

# create directories
savedir_labels = savedir + '/labels'
savedir_inputs = savedir + '/inputs'
create_dir(savedir)
create_dir(savedir_labels)
create_dir(savedir_inputs)

# for label 5 parameters create distribution histos
for i in range(0,5):
    name = "Plot_"+str(i)
    print_dist(savedir_labels,name,labels[:,i])

# calculate total inputs = weights times parameters
for i in range(0,12):
    for j in range(0,5):
#        print(j,i*6,i*6+j+1)
        inputs_total[:,j]=inputs_total[:,j]+inputs[:,i*6]*inputs[:,i*6+j+1]

# for 5 parameters create distribution histos
for i in range(0,5):
    name = "Plot_"+str(i)
    print_dist(savedir_inputs,name,inputs_total[:,i])

# checks
#diff = np.mean(np.transpose(labels-inputs_total),1)
#print('diff', diff)
#diff_means = np.mean(np.transpose(labels),1) - np.mean(np.transpose(inputs_total),1)
#print('diff_means', diff_means)
#print(np.mean(np.transpose(labels),1)/diff_means)
#print(np.mean(np.transpose(inputs_total),1)/diff_means)

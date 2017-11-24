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

#    print('Training set', train_dataset.shape, train_targets.shape)
#    print('Validation set', valid_dataset.shape, valid_targets.shape)
#    print('Test set', test_dataset.shape, test_targets.shape)

    total_dataset = train_dataset.shape[0] + valid_dataset.shape[0] + test_dataset.shape[0]
    input_heigth = train_dataset.shape[1]
    target_heigth = train_targets.shape[1]
    rearranged_length = total_dataset*12

    train_size = train_dataset.shape[0]
    valid_size = valid_dataset.shape[0]
    test_size = test_dataset.shape[0]

#    print('total dataset length', total_dataset)
#    print('input_heigth', input_heigth)
#    print('target_heigth', target_heigth)
#    print('train_size', train_size)
#    print('valid_size', valid_size)
#    print('test_size', test_size)

    inputs = np.zeros(shape=(total_dataset,input_heigth))
    inputs_total = np.zeros(shape=(total_dataset,5))
    inputs_rearranged = np.zeros(shape=(rearranged_length,5))
    targets = np.zeros(shape=(total_dataset,target_heigth))

#    print('inputs.shape', inputs.shape)
#    print('targets.shape', targets.shape)

# write split datasets into single dataset, save for targets
    inputs[0:train_size,:] = train_dataset
    targets[0:train_size,:] = train_targets
    inputs[train_size:train_size+valid_size,:] = valid_dataset
    targets[train_size:train_size+valid_size,:] = valid_targets
    inputs[train_size+valid_size:valid_size+train_size+test_size,:] = test_dataset
    targets[train_size+valid_size:valid_size+train_size+test_size,:] = test_targets

    return inputs,inputs_total,targets,inputs_rearranged,total_dataset

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

inputs,inputs_total,targets,inputs_rearranged,total_dataset = get_data()

# create directories
savedir_targets = savedir + '/targets'
savedir_inputs = savedir + '/inputs'
savedir_inputs_total = savedir + '/inputs_total'
create_dir(savedir)
create_dir(savedir_targets)
create_dir(savedir_inputs)
create_dir(savedir_inputs_total)

# for target 5 parameters create distribution histos
for i in range(0,5):
    name = "Plot_"+str(i)
    print_dist(savedir_targets,name,targets[:,i])


# calculate total inputs = weights times parameters
for k in range(0,total_dataset):
    for i in range(0,12):
        for j in range(0,5):
#            print(k*12+i,j,"",int(round(k/12)),i*6+j+1)
            inputs_rearranged[k*12+i,j]=inputs[int(round(k/12)),i*6+j+1]

# for 60 compontents = 12*5 reco parameters create distribution histos
for i in range(0,5):
    name = "Plot_"+str(i)
    print_dist(savedir_inputs,name,inputs_rearranged[:,i])


# calculate total inputs = weights times parameters
for i in range(0,12):
    for j in range(0,5):
#        print(j,i*6,i*6+j+1)
        inputs_total[:,j]=inputs_total[:,j]+inputs[:,i*6]*inputs[:,i*6+j+1]

# for 5 total input reco parameters create distribution histos
for i in range(0,5):
    name = "Plot_"+str(i)
    print_dist(savedir_inputs_total,name,inputs_total[:,i])

# checks
#diff = np.mean(np.transpose(targets-inputs_total),1)
#print('diff', diff)
#diff_means = np.mean(np.transpose(targets),1) - np.mean(np.transpose(inputs_total),1)
#print('diff_means', diff_means)
#print(np.mean(np.transpose(targets),1)/diff_means)
#print(np.mean(np.transpose(inputs_total),1)/diff_means)

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle

savedir = "distribution_histo_plots/analysis"

#-------------------------------------------------------------------------------
# functions


def get_data():
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

#    print('Training set', train_dataset.shape, train_targets.shape)
#    print('Validation set', valid_dataset.shape, valid_targets.shape)
#    print('Test set', test_dataset.shape, test_targets.shape)

    total_dataset = train_dataset.shape[0] + \
        valid_dataset.shape[0] + test_dataset.shape[0]
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

    inputs = np.zeros(shape=(total_dataset, input_heigth))
    inputs_total = np.zeros(shape=(total_dataset, 5))
    inputs_rearranged = np.zeros(shape=(rearranged_length, 5))
    targets = np.zeros(shape=(total_dataset, target_heigth))

#    print('inputs.shape', inputs.shape)
#    print('targets.shape', targets.shape)

# write split datasets into single dataset, save for targets
    inputs[0:train_size, :] = train_dataset
    targets[0:train_size, :] = train_targets
    inputs[train_size:train_size+valid_size, :] = valid_dataset
    targets[train_size:train_size+valid_size, :] = valid_targets
    inputs[train_size+valid_size:valid_size +
           train_size+test_size, :] = test_dataset
    targets[train_size+valid_size:valid_size +
            train_size+test_size, :] = test_targets

    return inputs, inputs_total, targets, inputs_rearranged, total_dataset


def create_dir(savedir):
    try:
        os.mkdir(savedir)
    except:
        print(savedir, "already created")


#-------------------------------------------------------------------------------
# main code


inputs, inputs_total, targets, inputs_rearranged, total_dataset = get_data()

# max analysis
tar = np.zeros(shape=(targets.shape))
for i in range(0, 5):
    tar[:, i] = np.sort(targets[:, i])

# save 100 largest params
create_dir(savedir)
create_dir(savedir + "/min")
create_dir(savedir + "/max")

for j in range(0, 5):
    f = open(savedir + "/min/paras_" + str(j+1) + "_anaylsis.csv", "w")
    for i in range(0, 100):
        f.write(str(tar[-i-1, j]))
        f.write('\n')
    f.close()

for j in range(0, 5):
    f = open(savedir + "/max/paras_" + str(j+1) + "_anaylsis.csv", "w")
    for i in range(0, 100):
        f.write(str(tar[i, j]))
        f.write('\n')
    f.close()

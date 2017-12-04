import numpy as np
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt

#savedir = "ml_output"
savedir = "ml_output_hidden"

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

    print('Training set', train_dataset.shape, train_targets.shape)
    print('Validation set', valid_dataset.shape, valid_targets.shape)
    print('Test set', test_dataset.shape, test_targets.shape)

    total_dataset = train_dataset.shape[0] + valid_dataset.shape[0] + test_dataset.shape[0]
    input_heigth = train_dataset.shape[1]
    target_heigth = train_targets.shape[1]

    train_size = train_dataset.shape[0]
    valid_size = valid_dataset.shape[0]
    test_size = test_dataset.shape[0]

    inputs = np.zeros(shape=(total_dataset,input_heigth))
    targets = np.zeros(shape=(total_dataset,target_heigth))

# write split datasets into single dataset, save for targets
    inputs[0:train_size,:] = train_dataset
    targets[0:train_size,:] = train_targets
    inputs[train_size:train_size+valid_size,:] = valid_dataset
    targets[train_size:train_size+valid_size,:] = valid_targets
    inputs[train_size+valid_size:valid_size+train_size+test_size,:] = test_dataset
    targets[train_size+valid_size:valid_size+train_size+test_size,:] = test_targets

# other definitions
#    logits = np.zeros(shape=(total_dataset,target_heigth))

#    return inputs,targets,logits
    return inputs,targets

def get_ml_data(savedir):
    logits = np.loadtxt(savedir + "/logits.csv", delimiter=",")
#    weights = np.loadtxt(savedir + "/weights.csv", delimiter=",")
#    biases = np.loadtxt(savedir + "/biases.csv", delimiter=",")

#    return weights,biases
    return logits

#def data_check(inputs,targets,weights,biases,logits):
def data_check(inputs,targets,logits):
    print('inputs.shape', inputs.shape)
    print('targets.shape', targets.shape)
#    print('weights.shape', weights.shape)
#    print('biases.shape', biases.shape)
    print('logits.shape', logits.shape)

def print_dist(savedir,name,plotdata):
    plt.figure(name)
    plt.title(name)
    plt.hist(plotdata, normed=True, bins=30)
#    plt.show()
    plt.savefig(savedir + '/' + name + '.png')
    plt.gcf().clear()

#-------------------------------------------------------------------------------
# main code

inputs,targets = get_data()

# get weights & biases from tensorflow ml script
logits = get_ml_data(savedir)

# data_check
#data_check(inputs,targets,weights,biases,logits)
data_check(inputs,targets,logits)

# calculate logits
#logits = np.matmul(inputs,weights) + biases

# calculate difference
diff = logits-targets

# for 2 target parameters create distribution histos
for i in range(0,targets.shape[1]):
    name = "Targets_Para_" + str(i+1)
    print_dist(savedir,name,targets[:,i])

# for 2 logits (ml_output) parameters create distribution histos
for i in range(0,logits.shape[1]):
    name = "ML_Output_Para_" + str(i+1)
    print_dist(savedir,name,logits[:,i])

# for 2 logits (ml_output) parameters create distribution histos
for i in range(0,logits.shape[1]):
    name = "Diff_Para_" + str(i+1)
    print_dist(savedir,name,diff[:,i])

# save logits & difference
np.savetxt("ml_output/logits.csv", logits, delimiter=",")
np.savetxt("ml_output/diff.csv", diff, delimiter=",")

print('programm terminated.')

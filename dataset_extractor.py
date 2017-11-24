import os
import csv
import numpy as np
from six.moves import cPickle as pickle

workingdir = 'ml_input_2017.11.24_22_08_29'

#===============================================================================
# functions

# get length of csv file
def getlength(filename):
    filelength = 0
    f = open(workingdir + '/' + filename, 'rb')
    csvread = csv.reader(f)
    for line in csvread:
        for item in line:
            filelength += 1
    return filelength

# get data of csv file
def getdata(array, arrayheight, filename):
    i = -1
    j = -1

    f = open(workingdir + '/' + filename, 'rb')
    csvread = csv.reader(f)
    for line in csvread:
#        print line
        for item in line:
            i += 1
            if (i % arrayheight == 0):
                j += 1
                i = 0
#            print i,j
#            print item
            array[i,j] = item

#    print array[0:10,0]
#    print array.shape
    return array

#randomize arrays
def randomize(dataset, targets):
  permutation = np.random.permutation(targets.shape[0])
  shuffled_dataset = dataset[permutation,:]
  shuffled_targets = targets[permutation,:]
  return shuffled_dataset, shuffled_targets

# spilt data set into training, validation and testing dataset

#===============================================================================
# main

# generate input and target arrays
inputlength = getlength('inputs.csv')
inputheigth = 72
inputs = np.zeros(shape=(inputheigth,inputlength/inputheigth))
inputs = getdata(inputs, inputheigth, 'inputs.csv')

targetlength = getlength('targets.csv')
targetheigth = 5
targets = np.zeros(shape=(targetheigth,targetlength/targetheigth))
targets = getdata(targets, targetheigth, 'targets.csv')

# transpose datasets
inputs = np.transpose(inputs)
targets = np.transpose(targets)

print "inputs shape =", inputs.shape
print "targets shape =", targets.shape
print ""


# randomize
inputs, targets = randomize(inputs, targets)

# define data sizes
#train_size = 200000
#valid_size = 10000
#test_size = 10000
test_size = inputlength/inputheigth/20
valid_size = test_size
train_size = inputlength/inputheigth-test_size*2

print "test_size =", test_size
print "train_size =", train_size
print "valid_size =", valid_size
print ""

# split dataset
train_dataset = inputs[0:test_size,:]
train_targets = targets[0:test_size,:]
valid_dataset = inputs[test_size:test_size+valid_size,:]
valid_targets = targets[test_size:test_size+valid_size,:]
test_dataset = inputs[test_size+valid_size:test_size+valid_size+train_size,:]
test_targets = targets[test_size+valid_size:test_size+valid_size+train_size,:]

# sizes
print('Training:', train_dataset.shape, train_targets.shape)
print('Validation:', valid_dataset.shape, valid_targets.shape)
print('Testing:', test_dataset.shape, test_targets.shape)
print ""


# save to pickle
pickle_file = os.path.join('data_root', '5para.pickle')

if (os.path.exists('data_root')==False):
    os.makedirs('data_root')
    print "path created"
else:
    print "data_root directory aleady exists"

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_targets': train_targets,
        'valid_dataset': valid_dataset,
        'valid_targets': valid_targets,
        'test_dataset': test_dataset,
        'test_targets': test_targets,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    print "datasets saved as pickle"
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

print ""

# length checks
print "difference between input length and dataset length =", inputlength/inputheigth - (len(train_dataset) + len(test_dataset) + len(valid_dataset))
print "difference between input length and targets length =", inputlength/inputheigth - (len(train_targets) + len(test_targets) + len(valid_targets))

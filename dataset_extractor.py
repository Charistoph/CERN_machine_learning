import os
import csv
import numpy as np
from six.moves import cPickle as pickle

workingdir = 'ml_input_2017.11.23_09_14_11'

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
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:]
  shuffled_labels = labels[permutation,:]
  return shuffled_dataset, shuffled_labels

# spilt data set into training, validation and testing dataset

#===============================================================================
# main

# generate input and label arrays
inputlength = getlength('inputs.csv')
inputheigth = 72
inputs = np.zeros(shape=(inputheigth,inputlength/inputheigth))
inputs = getdata(inputs, inputheigth, 'inputs.csv')

labellength = getlength('labels.csv')
labelheigth = 5
labels = np.zeros(shape=(labelheigth,labellength/labelheigth))
labels = getdata(labels, labelheigth, 'labels.csv')

# transpose datasets
inputs = np.transpose(inputs)
labels = np.transpose(labels)

print "inputs shape =", inputs.shape
print "labels shape =", labels.shape
print ""


# randomize
inputs, labels = randomize(inputs, labels)

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
train_labels = labels[0:test_size,:]
valid_dataset = inputs[test_size:test_size+valid_size,:]
valid_labels = labels[test_size:test_size+valid_size,:]
test_dataset = inputs[test_size+valid_size:test_size+valid_size+train_size,:]
test_labels = labels[test_size+valid_size:test_size+valid_size+train_size,:]

# sizes
print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)
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
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
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
print "difference between input length and labels length =", inputlength/inputheigth - (len(train_labels) + len(test_labels) + len(valid_labels))

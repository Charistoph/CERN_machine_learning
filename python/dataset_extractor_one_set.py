import os
import csv
import numpy as np
from six.moves import cPickle as pickle

workingdir = 'ml_input/2018.10.12_18.25.31'
# workingdir = '/Users/christoph/Documents/coding/CERN_input_data/python/2018.10.12_18.25.31'
data_root_path = 'data_root'
# data_root_path = '/Users/christoph/Documents/coding/CERN_input_data/python/data_root'

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
            array[i, j] = item

#    print array[0:10,0]
#    print array.shape
    return array

# randomize arrays


def randomize(dataset, targets):
    permutation = np.random.permutation(targets.shape[0])
    shuffled_dataset = dataset[permutation, :]
    shuffled_targets = targets[permutation, :]
    return shuffled_dataset, shuffled_targets

# spilt data set into training, validation and testing dataset

# ===============================================================================
# main


#### ---- Marker 1 ---- ####
# generate input and target arrays
inputlength = getlength('inputs.csv')
inputheight = 72
inputs = np.zeros(shape=(inputheight, inputlength/inputheight))
inputs = getdata(inputs, inputheight, 'inputs.csv')

targetlength = getlength('targets.csv')
targetheight = 5
targets = np.zeros(shape=(targetheight, targetlength/targetheight))
targets = getdata(targets, targetheight, 'targets.csv')


#### ---- Marker 2 ---- ####
# transpose datasets
inputs = np.transpose(inputs)
targets = np.transpose(targets)

print "inputs shape =", inputs.shape
print "targets shape =", targets.shape
print ""


#### ---- Marker 3 ---- ####
# randomize
inputs, targets = randomize(inputs, targets)

# sizes
print('Training:', inputs.shape, targets.shape)
print ""


#### ---- Marker 5 ---- ####
# save to pickle
pickle_file = os.path.join(data_root_path, '5para_one_set.pickle')

if (os.path.exists(data_root_path) == False):
    os.makedirs(data_root_path)
    print "path created"
else:
    print "data_root directory aleady exists"

try:
    f = open(pickle_file, 'wb')
    save = {
        'inputs': inputs,
        'targets': targets,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    print "datasets saved as pickle"
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

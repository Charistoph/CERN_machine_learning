import os
import csv
import numpy as np

workingdir = 'ml_input_2017.11.13_13_02_26'

#===============================================================================
# functions
def getlength(filename):
    filelength = 0
    f = open(workingdir + '/' + filename, 'rb')
    csvread = csv.reader(f)
    for line in csvread:
        for item in line:
            filelength += 1
    return filelength

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

#===============================================================================
# main

# generate input and label arrays
inputlength = getlength('inputs.csv')
inputheigth = 72
inputs = np.zeros(shape=(inputheigth,inputlength/inputheigth))
inputs = getdata(inputs, inputheigth, 'inputs0.csv')

labellength = getlength('labels.csv')
labelheigth = 5
label = np.zeros(shape=(labelheigth,labellength/labelheigth))
label = getdata(label, labelheigth, 'labels.csv')

from random import seed
from random import randrange
from csv import reader
from math import sqrt
import csv
import numpy  as np
# Load a CSV file
def load_csv(filename):
	file = open(filename,  "rt")
	lines = reader(file)
	dataset = list(lines)
	return dataset
'''
def alpha_to_num(labels,outputs):
    
    output_form = np.zeros((len(outputs),len(labels)))
    for i in range(len(outputs)):
		index = outputs[i]
		ind   = label[index[0]]
		output_form[i][ind] = 1
	return output_form
'''
def alpha_to_num(labels,outputs):
    output_form = np.zeros((len(outputs),len(labels)))
    for i in range(len(outputs)):
        index = outputs[i]
        ind = label[index[0]]
        output_form[i][ind] = 1
    return output_form

'''
data = load_csv("activity/train.csv")
data = np.array(data[1:])
data = np.delete(data,[len(data[0])-1,len(data[0])-2],1)
data = data.astype(np.float64)
np.savetxt("train.csv", data, delimiter=",",fmt="%f")
'''



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = x.transpose()
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum(axis=0)+ 1e-15).transpose()
def soft(x):
    """Compute softmax values for each sets of scores in x."""
   
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum(axis=0)+ 1e-15)


data = load_csv("activity/train.csv")
data = np.array(data[1:])

np.random.shuffle(data)
outputString = data[:,[len(data[0])-1]]

data = np.delete(data,[len(data[0])-1,len(data[0])-2],1)
data = data.astype(np.float64)

np.savetxt("train.csv", data, delimiter=",",fmt="%f")


label = {'WALKING':0,'WALKING_UPSTAIRS':1,'WALKING_DOWNSTAIRS':2,'SITTING':3,'STANDING':4,'LAYING':5}

#change labels to output arrays
output = alpha_to_num(label,outputString)

np.savetxt("trainTarget.csv", output, delimiter=",",fmt="%f")

print(data.shape, "    " , output.shape)


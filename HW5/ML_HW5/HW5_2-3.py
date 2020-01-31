from svmutil import *
import csv
import numpy as np
import scipy.spatial

def linear_plus_rbf(xa,xb):
   rbf_kernel    = np.exp(-0.03*scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean'))
   linear_kernel = xa.dot(xb.T)
   return rbf_kernel+linear_kernel

train_img = []
with open('X_train.csv') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		dictOfimgs = [float(row[i]) for i in range(0, len(row))]
		train_img.append(dictOfimgs)

train_img = np.array(train_img)

train_label = []
with open('Y_train.csv') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		train_label.append(float(row[0]))
train_label = np.array(train_label)

train_batch = []
train_kernel = linear_plus_rbf(train_img,train_img)
count = 1
for row in train_kernel:
    dictOfimgs = { i+1 : float(row[i]) for i in range(0, len(row)) }
    dictOfimgs.update( {0 : count} )
    train_batch.append(dictOfimgs)
    count+=1

m = svm_train(train_label, train_batch, '-t 4 -c 1000')
del train_kernel
del dictOfimgs
test_img = []
with open('X_test.csv') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		dictOfimgs = [float(row[i]) for i in range(0, len(row))]
		test_img.append(dictOfimgs)
test_img = np.array(test_img)

test_batch = []
test_kernel = linear_plus_rbf(test_img,train_img)
test_img =[]
train_img = []
count = 1

for row in test_kernel:
    dictOfimgs = { i+1 : float(row[i]) for i in range(0, len(row)) }
    dictOfimgs.update( {0 : count} )
    test_batch.append(dictOfimgs)
    count+=1

test_label = []
with open('Y_test.csv') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		test_label.append(float(row[0]))

p_label, p_acc, p_val = svm_predict(test_label, test_batch , m)
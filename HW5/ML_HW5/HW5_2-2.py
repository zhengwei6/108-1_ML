from svmutil import *
import csv
import numpy as np

def grid_search(train_data, train_label, test_data, test_label):
    c = [100]
    g = [0.03]
    for i in c:
    	for j in g:
    		m = svm_train(train_label, train_data,'-s 0 -t 0 -c %s -g %s -q'%(i,j))
    		p_label, p_acc, p_val = svm_predict(test_label,test_data , m)


def cross_validaiton(train_img,train_label,K):
	#spilt the data into k fold
    fold_size = len(train_img)/K
    batches = []
    for i in range(K):
        print("------------------")
        all_index  = set(range(len(train_img)))
        test_index = set(range(int(i*fold_size), int((i+1)*fold_size)))
        train_index = all_index - test_index

        train_data_sub  = [train_img[i] for i in train_index]
        train_label_sub = [train_label[i] for i in train_index]
        test_data_sub  = [train_img[i] for i in test_index]
        test_label_sub = [train_label[i] for i in test_index]
        grid_search(train_data_sub, train_label_sub, test_data_sub, test_label_sub)

train_img = []
with open('X_train.csv') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		dictOfimgs = { i+1 : float(row[i]) for i in range(0, len(row)) }
		train_img.append(dictOfimgs)

train_label = []

with open('Y_train.csv') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		train_label.append(float(row[0]))

test_img = []
with open('X_test.csv') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		dictOfimgs = { i+1 : float(row[i]) for i in range(0, len(row)) }
		test_img.append(dictOfimgs)

test_label = []
with open('Y_test.csv') as csvFile:
	rows = csv.reader(csvFile, delimiter=',')
	for row in rows:
		test_label.append(float(row[0]))


random_state = np.random.get_state()
np.random.shuffle(train_img)
np.random.set_state(random_state)
np.random.shuffle(train_label)
cross_validaiton(train_img,train_label,10)
from svmutil import *
import csv


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

c = [0.1, 1, 10 , 100, 1000]
g = [0.001, 0.01, 0.02,0.03, 0.1]

for i in c:
	for j in g:
		m = svm_train(train_label, train_img,'-s 0 -t 2 -c %s -g %s -q'%(i,j))
		p_label, p_acc, p_val = svm_predict(test_label,test_img , m)


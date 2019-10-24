# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:26:53 2019

@author: Willy
"""
import numpy as np
from parse_dataset import *


TEST_IMG_NUM = 10000

def find_bins(value):
    return int(value/8)

def compute_log_likelihood(variance,mu,x):
    if variance != 0:
        return  -1/2 * np.log(2*np.pi) - 1/2 * np.log(variance) - 1/2 * (np.square(x - mu) )/ variance
    elif x == mu:
        return 1
    else:
        return 0

class naive_bayes_classifier(object): 
    def __init__(self, train_img_path, train_label_path, mode=1):
        train_img_array,train_label_array = parse_datasets(train_img_path,train_label_path)
        # compute prior
        self.prior = np.array([np.mean(train_label_array == l) for l in range(10)])
        
        # compute label frequency
        unique, counts  = np.unique(train_label_array, return_counts=True)
        self.label_count = dict(zip(unique, counts))
        self.disc_lut    = np.zeros([10,28,28,32]) # label index gray level
        self.label_variance = np.zeros([10,28,28,1])
        self.label_mean     = np.zeros([10,28,28,1])
        if mode == 1:
            # count for the gray level
            for index in range(train_img_array.shape[0]):
                for i in range(28):
                    for j in range(28):
                        self.disc_lut[ int(train_label_array[index,0]) , i , j , find_bins(train_img_array[index,i,j]) ] += 1
        else:
            # compute mean
            for index in range(train_img_array.shape[0]):
                for i in range(28):
                    for j in range(28):
                        self.label_mean[int(train_label_array[index,0]), i , j , 0] += train_img_array[index,i,j]
            for label_index in range(10):
                self.label_mean[label_index] = self.label_mean[label_index] / self.label_count[label_index]
            # compute variance
            for index in range(train_img_array.shape[0]):
                label = int(train_label_array[index,0])
                for i in range(28):
                    for j in range(28):
                        self.label_variance += np.square(train_img_array[label,i,j] - self.label_mean[label,i,j,0])
            for label_index in range(10):
                self.label_variance = self.label_variance / self.label_count[label_index]
    
    def _display_label_prob(self,label_prob,label_ans):
        label_prob       = np.abs(label_prob)
        label_prob       = label_prob / label_prob.sum()
        label_prediction = np.argmin(label_prob)
        print("Postirior (in log scale):")
        for i in range(10):
            print(i,":",label_prob[i])
        print("Prediciton:",label_prediction,", Ans:",label_ans)
        print("")
        return label_prediction 
    
    def dicrete_prediction_postier(self,test_img_path, test_label_path):
        test_img_array,test_label_array = parse_datasets(test_img_path,test_label_path)
        correct_count = 0.0
        for index in range(TEST_IMG_NUM):
            label_prob = np.zeros([10])
            for label in range(10):
                total_sum = 0.0
                for i in range(28):
                    for j in range(28):
                        bins = find_bins(test_img_array[index,i,j])
                        if self.disc_lut[label,i,j,bins] == 0:
                            self.disc_lut[label,i,j,bins] = self.disc_lut[label,i,j,np.where(self.disc_lut[label,i,j]>0)].min()
                        total_sum += np.log(self.disc_lut[label,i,j,bins]) - np.log(self.label_count[label])
                total_sum = total_sum + self.prior[label]
                label_prob[label] = total_sum
            prediction_ans = self._display_label_prob(label_prob,test_label_array[index,0])
            if prediction_ans == test_label_array[index,0]:
                if index % 1000 == 0:
                    print(correct_count/(index+1))
                correct_count += 1
        print("Error rate:",1-correct_count/TEST_IMG_NUM)
        return

    def continous_prediction_postier(self,test_img_path, test_label_path):
        test_img_array,test_label_array = parse_datasets(test_img_path,test_label_path)
        correct_count = 0.0
        for index in range(TEST_IMG_NUM):
           label_prob = np.zeros([10])
           for label in range(10):
               total_sum = 0.0
               for i in range(28):
                   for j in range(28):
                       total_sum += compute_log_likelihood(self.label_variance[label,i,j,0], self.label_mean[label,i,j,0], test_img_array[index,i,j])
               total_sum = total_sum + np.log(self.prior[label])
               label_prob[label] = total_sum
           prediction_ans = self._display_label_prob(label_prob,test_label_array[index,0])
           if prediction_ans == test_label_array[index,0]:
               if index % 1000 == 0:
                   print(correct_count/(index+1))
               correct_count += 1
        print("Error rate:",1-correct_count/TEST_IMG_NUM)
        return

    def display_imagination(self):
        print("Imagination of numbers in Bayesian classifier:")
        for label in range(10):
            print(label,":")
            for i in range(28):
                for j in range(28):
                    if self.label_mean[label,i,j] < 128:
                        print(0,end=' ')
                    else:
                        print(1,end=' ')
                print("")
            print("")
            
if __name__ == "__main__":
    bayes_classifier = naive_bayes_classifier("train-images.idx3-ubyte","train-labels.idx1-ubyte",mode=1)
    bayes_classifier.dicrete_prediction_postier("t10k-images.idx3-ubyte","t10k-labels.idx1-ubyte")
    bayes_classifier = naive_bayes_classifier("train-images.idx3-ubyte","train-labels.idx1-ubyte",mode=2)
    bayes_classifier.continous_prediction_postier("t10k-images.idx3-ubyte","t10k-labels.idx1-ubyte")
    bayes_classifier.display_imagination()
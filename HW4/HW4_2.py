# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 21:57:45 2019

@author: Willy
"""
import numpy as np
from parse_dataset import *
import random
from matplotlib import pyplot as plt
import random
from datetime import datetime
import time

random.seed(datetime.now())
class EM_aglorithm(object): 
    def __init__(self, images, labels, sample_num):
        self.class_matrix = np.zeros([10,10])
        self.img   = images
        self.label = labels
        self.clas  = 10
        self.sample_num = sample_num
        random.seed(time.time())
        self.prob = np.array([random.uniform(0.25,0.75) for i in range(self.img.shape[0] * self.img.shape[1] * self.clas)])
        self.prob = self.prob.reshape((28, 28, self.clas))
        self.prob[:,:] = self.prob[:,:] / np.sum(self.prob, axis=(0,1))
        self.copy_matrix = self.prob
        self.lamda  = np.array([1. / self.sample_num for i in range(self.sample_num)])
        
    def class_proba(self):
        n_imgs = self.img.shape[2]
        p_c = np.ones((n_imgs, self.clas))
        for n in range(n_imgs):
            for c in range(self.clas):
                p_c[n, c] = np.nanprod(np.multiply(np.power(self.prob[:,:, c], self.img[:,:,n]), np.power(1 - self.prob[:, :, c], 1 - self.img[:,:,n]))) * self.lamda[c]
        return p_c
    
    def responsability(self, p_c): 
        w = p_c
        marginal = np.sum(w, axis=1)
        for c in range(self.clas): 
            w[:, c] /= marginal
        return np.nan_to_num(w)
    
    def update_params(self, w): 
        self.lamda = np.sum(w, axis=0) / self.sample_num
        for c in range(self.clas):
            self.prob[:, :, c] = w[0, c] * self.img[:,:, 0]
            for n in range(1, self.sample_num): 
                self.prob[:, :,c] += w[n, c] * self.img[:,:, n]
        self.prob /= self.lamda * self.sample_num
    
    def run_once(self):
        p_c = self.class_proba()
        w = self.responsability(p_c)
        self.update_params(w)
        return np.copy(self.lamda), np.copy(self.prob)
    
    
def image2bin(img_array):
    bin_img_array = np.zeros(img_array.shape)
    for k in range(img_array.shape[0]):
        for i in range(img_array.shape[1]):
            for j in range(img_array.shape[2]):
                if img_array[k,i,j] >= 127 :
                    bin_img_array[k,i,j] = 1.0
                else:
                    bin_img_array[k,i,j] = 0.0
    return bin_img_array

def ouput_class_prob(orignal_prob, bin_img_array, label_array, lamda):
    class_count = np.zeros([10,10])
    n_imgs = bin_img_array.shape[2]
    p_c = np.ones((n_imgs, 10))
    for n in range(n_imgs):
        for c in range(10):
            p_c[n, c] = np.nanprod(np.multiply(np.power(orignal_prob[:,:, c], bin_img_array[:,:,n]), np.power(1 - orignal_prob[:, :, c], 1 - bin_img_array[:,:,n]))) * lamda[c]
    marginal = np.sum(p_c, axis=1)
    for c in range(10): 
        p_c[:, c] /= marginal
    for i in range(n_imgs):
        posible_class = np.argmax(p_c[i,:])
        class_count[int(label_array[i]),int(posible_class)] += 1
    return class_count
    
def change_class_prob(class_count,orignal_prob):
    change_prob = np.zeros(orignal_prob.shape)
    confid_rate = np.zeros([10])
    bool_index  = np.zeros([10])
    # compute confid rate
    for i in range(class_count.shape[0]):
        confid_rate[i] = np.amax(class_count[i,:]) / np.sum(class_count[i,:])
    iterate = np.argsort(confid_rate)
    
    for i in iterate[::-1]:
        change_index = np.argmax(class_count[i,:])
        while bool_index[change_index] == 1:
            class_count[i,change_index] = -1
            change_index = np.argmax(class_count[i,:])
        bool_index[change_index] = 1
        print(change_index)
        change_prob[:,:,i] = orignal_prob[:,:,change_index]
    return change_prob

def confusion_matrix(prob,bin_img_array,label_array,lamda):
    #create confusion matrix
    confusion_matrix = np.zeros([10,10])
    n_imgs = bin_img_array.shape[2]
    p_c = np.ones((n_imgs, 10))
    for n in range(n_imgs):
        for c in range(10):
            p_c[n, c] = np.nanprod(np.multiply(np.power(prob[:,:, c], bin_img_array[:,:,n]), np.power(1 - prob[:, :, c], 1 - bin_img_array[:,:,n]))) * lamda[c]
    marginal = np.sum(p_c, axis=1)
    for c in range(10): 
        p_c[:, c] /= marginal
    for i in range(n_imgs):
        posible_class = np.argmax(p_c[i,:])
        confusion_matrix[int(posible_class),int(label_array[i])] += 1
    return confusion_matrix

def error_rate(confusion_matrix):
    for i in range(10):
        print("Confusion Matrix ",i)
        pred_not_sum_wrong = 0
        pred_not_sum = 0
        for k in range(10):
            if k != i:
                pred_not_sum_wrong += confusion_matrix[k,i]
        for k in range(10):
            if k != i:
                pred_not_sum += np.sum(confusion_matrix[k,:])
        print("Sensitivity", end=' ')
        print(confusion_matrix[i,i] / np.sum(confusion_matrix[i,:]))
        print("Specificity",end=' ')
        print(1 - (pred_not_sum_wrong / pred_not_sum))
    return

if __name__ == '__main__':
    train_img_array,train_label_array = parse_datasets("train-images.idx3-ubyte","train-labels.idx1-ubyte")
    #test_img_array,test_label_array   = parse_datasets("t10k-images.idx3-ubyte","t10k-labels.idx1-ubyte")
    train_img_array = train_img_array[:,:,:60000]
    train_label_array = train_label_array[:60000]
    bin_img_array = image2bin(train_img_array)
    em = EM_aglorithm(bin_img_array, train_label_array, 60000)
    

    lb_old = 0
    pr_old = 0
    for i in range(20):
        lb,pr  = em.run_once()
        delta_lb = np.linalg.norm(lb - lb_old)
        delta_pr = np.linalg.norm(pr - pr_old)
        if delta_lb < 0.000001 and delta_pr < 0.0001: 
            break
        print("update lb : %f"%(delta_lb))
        print("update pr : %f"%(delta_pr))
        lb_old = lb
        pr_old = pr
    class_count = ouput_class_prob(em.prob,bin_img_array,train_label_array,em.lamda)
    change_prob = change_class_prob(class_count,em.prob)
    confusion_matrix = confusion_matrix(change_prob,bin_img_array,train_label_array,em.lamda)
    error_rate(confusion_matrix)
    plt.figure(1)
    for i in range(2):
        for j in range(5):
            plt.subplot(2,5,i*5+j+1)
            plt.imshow(change_prob[:,:,i*5+j])
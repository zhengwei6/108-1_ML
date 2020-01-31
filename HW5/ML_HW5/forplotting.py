# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 14:16:12 2019

@author: Willy
"""
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
x = [1,2,3,4]
labels = ['Linear', 'Polynomial', 'RBF', 'RBF+Polynomail']
test_time_acc = [96,97.76,98.56,95.68]
train_time_acc = [96.2,98.6,98.9,95.8]
fig,ax1 = plt.subplots()
ax1.set_title('different kernel performance');
ax1.plot(labels,test_time_acc, label='test')
ax1.plot(labels,train_time_acc, label='train')
ax1.legend()


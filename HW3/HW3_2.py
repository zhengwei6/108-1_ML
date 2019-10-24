# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 10:21:29 2019

@author: Willy
"""
from HW3_1 import *
def online_sequential_stimator(data_list,mu,variance):
    # produce new data point
    new_data_point = box_muller(3.0,5.0)
    print("Add data point: ",new_data_point[0])
    data_list.extend(box_muller(3.0,5.0))
    
    # compute new mu & new variance
    new_mu = mu + (new_data_point - mu) / (len(data_list))
    new_variance = variance + ((new_data_point - mu)*(new_data_point - new_mu) - variance) / (len(data_list))
    print('Mean = {:f}  Variance = {:f}'.format(float(new_mu),float(new_variance)))
    if np.abs(new_mu - mu) < 0.001 and np.abs(new_variance - variance) < 0.001:
        return
    else:
        online_sequential_stimator(data_list,new_mu,new_variance)

if __name__ == '__main__':
    print("Data point source function: N(3.0, 5.0)")
    data_list = []
    mu  = 3.0
    variance = 5.0
    online_sequential_stimator(data_list,mu,variance)
    
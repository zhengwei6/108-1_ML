# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 23:23:50 2019

@author: Willy
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.spatial
from numpy import linalg as LA
import random
k = 2
def Rbf_kernel(data1, data2, sigma):
    result = np.exp( -sigma * scipy.spatial.distance.cdist(data1, data2, 'sqeuclidean') )
    return result

def custom_kernel(s1, s2, c1, c2, sigma1, sigma2):
    term1 = Rbf_kernel(s1, s2, sigma1)
    term2 = Rbf_kernel(c1, c2, sigma2)
    return  term1*term2 

img = mpimg.imread('./image1.png')
img    = img.reshape(-1,3)
input_size   = img.shape[0]
data_cluster = {i:[] for i in range(k)}


for i in range(10000):
    index = random.randint(0, k-1) 
    data_cluster[index].append(i)

pixel_axis   = []
for i in range(100):
    for j in range(100):
        temp = [j/100, 1-i/100]
        pixel_axis.append(temp)
pixel_axis = np.array(pixel_axis).reshape(-1,2)

distance_matrix = np.asarray([[] for i in range(k)])
count = 0

while(1):
    distance_matrix = np.zeros((k,10000))
    for i in range(k):
        cluster_pixel_axis = pixel_axis[data_cluster[i]]
        cluster_img        = img[data_cluster[i]]
        second_term_value  = custom_kernel(pixel_axis,cluster_pixel_axis , img , cluster_img, 0.5,0.5).sum(axis=1) / cluster_pixel_axis.shape[0]
        third_term_value   = custom_kernel(cluster_pixel_axis,cluster_pixel_axis,cluster_img,cluster_img,0.5,0.5).sum(axis=1) / cluster_pixel_axis.shape[0]**2
        third_term_value   = third_term_value.sum(axis=0)        
        distance_matrix[i] = -2*second_term_value + third_term_value
    #assign
    belong_cluster = np.argmin(distance_matrix, axis=0)
    old_data_cluster = data_cluster
    data_cluster = {i:[] for i in range(k)}
    for i in range(10000):
        data_cluster[belong_cluster[i]].append(i)
    converge = 0
    for i in range(k):
        new_c = set(data_cluster[i])
        org_c = set(old_data_cluster[i])
        if new_c != org_c:
            converge = 1
    #plot new image
    for i in range(k):
        plt.scatter(pixel_axis[data_cluster[i],0],pixel_axis[data_cluster[i],1])
    plt.title(count)
    count += 1
    plt.savefig(str(count)+'.png')
    plt.pause(.1)
    plt.show()
    if converge == 0:
        break
    plt.cla()



    








    
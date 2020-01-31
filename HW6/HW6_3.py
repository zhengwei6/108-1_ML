# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 19:18:41 2019
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

def buildSimilarityGraph(img):
    pixel_axis   = []
    for i in range(100):
        for j in range(100):
            temp = [j/100, 1-i/100]
            pixel_axis.append(temp)
    pixel_axis = np.array(pixel_axis).reshape(-1,2)
    s_term = Rbf_kernel(img, img, 0.5)
    t_term = Rbf_kernel(pixel_axis, pixel_axis, 0.5)
    new_kernel = s_term * t_term
    return new_kernel
    
def buildDegreeMatrix(similarityMatrix):
    result = np.diag(similarityMatrix.sum(axis=1).ravel())
    return result

def unnormalizedLaplacian(simMatrix, degMatrix):
    result = degMatrix - simMatrix
    return result

def find_eig(laplacian):
    global k
    e_vals, e_vecs = LA.eig(np.matrix(laplacian))
    return e_vals, e_vecs

e_vals = 0
e_vecs = 0
img    = mpimg.imread('./image2.png')
img    = img.reshape(-1,3)

pixel_axis   = []
for i in range(100):
    for j in range(100):
        temp = [j/100, 1-i/100]
        pixel_axis.append(temp)
    
pixel_axis = np.array(pixel_axis).reshape(-1,2)
SimilarityGraph  = buildSimilarityGraph(img)
degMatrix   = buildDegreeMatrix(SimilarityGraph)
lapMatrix   = unnormalizedLaplacian(SimilarityGraph, degMatrix)

#normalize cut
L = lapMatrix
eigval, eigvec = np.linalg.eigh(L)
eigen = sorted([(eigval[i], eigvec[:,i]) for i in range(len(eigval))], key=lambda t:t[0]) 

U = np.ndarray((10000, k))
for i in range(1,k+1):
    U[:,i - 1] = eigen[i][1]
transform_data = U
#initial cluster
data_cluster = {i:[] for i in range(k)}
for i in range(10000):
    index = random.randint(0, k-1) 
    data_cluster[index].append(i)

centeriod    = np.array([transform_data[random.randint(0, img.shape[0])]  for i in range(k) ]).reshape(-1,k)

count = 0

#find the center point
while(1):
    print(centeriod)
    distance_matrix = np.zeros((k,10000))
    for i in range(k):
        temp = scipy.spatial.distance.cdist(transform_data, [centeriod[i]], 'sqeuclidean')
        distance_matrix[i] = np.transpose(temp)
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
    if converge == 0:
        break
    #plot new image
    for i in range(k):
        plt.scatter(pixel_axis[data_cluster[i],0],pixel_axis[data_cluster[i],1])
    plt.title(count)
    count += 1
    plt.savefig(str(count)+'.png')
    plt.pause(.1)
    plt.show()
    plt.cla()
    #update centeriod
    for i in range(k):
        centeriod[i] = np.mean(transform_data[data_cluster[i]],axis=0)
    print("---------------")

  
    
    
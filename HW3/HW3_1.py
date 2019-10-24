# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 22:57:35 2019

@author: Willy
"""
import numpy as np
import matplotlib.pyplot as plt


def box_muller(mu,variance):
    u1 = np.random.uniform(0,1,1)
    u2 = np.random.uniform(0,1,1)
    theta = 2 * np.pi * u1
    r = np.sqrt(-2 * np.log(u2))
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return  x*np.sqrt(variance)+mu

def poly_linear_data_generator(n,a,w,x):
    phi = np.ones([w.shape[0]])
    accumulation = 1.0
    for i in range(w.shape[0]):
        phi[i] =  accumulation * w[i]
        accumulation = accumulation * x
    # compute e
    sum_value = phi.sum()
    print(sum_value)
    sum_value += box_muller(0,a)
    return sum_value
    
def main():
    poly_linear_data_generator(3 ,5.0 ,np.array([1,2,3]) ,0.5 )
    
if __name__ == '__main__':
    main()
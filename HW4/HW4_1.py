# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 21:24:08 2019

@author: Willy
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def box_muller(mean,variance):
    u1 = np.random.uniform(0,1,1)
    u2 = np.random.uniform(0,1,1)
    theta = 2 * np.pi * u1
    r = np.sqrt(-2 * np.log(u2))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return  (x*np.sqrt(variance) + mean)[0] 

class logistic_regression():
    def __init__(self, N, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2):
        #generate D1
        self.D = np.zeros([N*2,3])
        self.G = np.zeros([N*2,1])
        plt.figure(figsize=(9, 6))
        plt.subplot(221)
        for i in range(N*2):
            if i < 50:
                self.G[i]   = 0
                self.D[i,0] = 1
                self.D[i,1] = box_muller(mx1, vx1)
                self.D[i,2] = box_muller(my1, vy1)
                plt.scatter(self.D[i,1],self.D[i,2],c='b')
            else:
                self.G[i]   = 1
                self.D[i,0] = 1
                self.D[i,1] = box_muller(mx2, vx2)
                self.D[i,2] = box_muller(my2, vy2)
                plt.scatter(self.D[i,1],self.D[i,2],c='r')
        self.lr = 1e-2
        self.w = np.ones([3,1])
        self.N = N
        self.params = 3
    def compute_weight_x(self,Xi):
        return np.exp( -(Xi[0]*self.w[0] + Xi[1]*self.w[1] + Xi[2]*self.w[2] ))
                      
    def weight_multiply_data(self):
        error = np.zeros([self.N*2,1])
        temp = 0
        for i in range(self.N*2):
            temp     = 1/(1+ self.compute_weight_x(self.D[i]) )
            error[i] = self.G[i] -  temp      
        return error
    
    def gredient_plot(self, t1):
        y_plot = np.zeros(t1.shape[0])
        for i in range(t1.shape[0]):
            y_plot[i] = (self.w[0] + self.w[1] * t1[i]) / -self.w[2]
        return y_plot
    
    def gredient_desent(self):
        self.w = np.ones([3,1])
        # to determine convergence
        old_w = 0.0
        for i in range(10001):
            error = self.weight_multiply_data()
            self.w += self.lr * np.transpose(self.D).dot(error)
            if np.abs(np.sum(self.w) - old_w) < 1e-3:
                break
            old_w = np.sum(self.w)
    
    def compute_sub_hessian(self,Xi):
        xw = 0 
        if -Xi.dot(self.w) <= -709:
            xw = -708
        else:
            xw = -Xi.dot(self.w)
        return np.exp(np.exp(xw)) / ( 1 + np.exp(xw)) **2
    
    def compute_hessian(self,X):
        hess = np.zeros((3, 3))
        d = np.zeros([self.N*2,self.N*2])
        for i in range(self.N*2):
            d[i,i] = self.compute_sub_hessian(X[i])
        hess = np.transpose(X).dot(d).dot(X)
        return hess
    
    def newton_desent(self):
        self.w = np.ones([3,1])
        old_w = 0.0
        for i in range(10010):    
            hessian   = self.compute_hessian(self.D)
            gradient_ch = np.dot(np.transpose(self.D),(self.weight_multiply_data()))
            self.w = self.w + inv(hessian).dot(gradient_ch)
            if np.abs(np.sum(self.w) - old_w) < 1e-7:
                break
    
def logistic_reggression(w,Xi):
    xw = 0 
    if -Xi.dot(w) <= -709:
        xw = -708
    else:
        xw = -Xi.dot(w)
    return 1 / (1 + np.exp(xw))
    
if __name__ == '__main__':
    
    x = logistic_regression(50, 1, 2, 1, 2, 10, 2, 10, 2)
    x.gredient_desent()
    predict_cluster1_correct = 0
    predict_cluster2_correct = 0
    plt.subplot(222)
    for i in range(x.D.shape[0]):
        ans = logistic_reggression(x.w, x.D[i,:])
        if ans >= 0.5:
            plt.scatter(x.D[i,1],x.D[i,2],c='r')
        elif ans < 0.5:
            plt.scatter(x.D[i,1],x.D[i,2],c='b')
            
        if ans >= 0.5 and x.G[i] == 1:
            predict_cluster1_correct += 1
        elif ans<0.5 and x.G[i] == 0:
            predict_cluster2_correct += 1
    print("Gradient descent:")
    print("w")
    print(x.w)
    print("Confusion Matrix:")
    print("Predict cluster 1 Predict cluster 2")
    print("Is cluster 1",end=' ')
    print(predict_cluster1_correct,end=' ')
    print(50-predict_cluster1_correct)
    print("Is cluster 2",end=' ')
    print(50-predict_cluster2_correct,end=' ')
    print(predict_cluster2_correct)
    print("Sensitivity (Successfully predict cluster 1):",end=' ')
    print(predict_cluster1_correct/50)
    print("Specificity (Successfully predict cluster 2):",end=' ')
    print(predict_cluster2_correct/50)
    print((predict_cluster1_correct + predict_cluster2_correct) / (x.D.shape[0]))
    
    
    print("----------------------------------------")
    x.newton_desent()
    predict_cluster1_correct = 0
    predict_cluster2_correct = 0
    plt.subplot(223)
    for i in range(x.D.shape[0]):
        ans = logistic_reggression(x.w, x.D[i,:])
        if ans >= 0.5:
            plt.scatter(x.D[i,1],x.D[i,2],c='r')
        elif ans < 0.5:
            plt.scatter(x.D[i,1],x.D[i,2],c='b')
            
        if ans >= 0.5 and x.G[i] == 1:
            predict_cluster1_correct += 1
        elif ans<0.5 and x.G[i] == 0:
            predict_cluster2_correct += 1
    print("Newton:")
    print("w")
    print(x.w)
    print("Confusion Matrix:")
    print("Predict cluster 1 Predict cluster 2")
    print("Is cluster 1",end=' ')
    print(predict_cluster1_correct,end=' ')
    print(50-predict_cluster1_correct)
    print("Is cluster 2",end=' ')
    print(50-predict_cluster2_correct,end=' ')
    print(predict_cluster2_correct)
    print("Sensitivity (Successfully predict cluster 1):",end=' ')
    print(predict_cluster1_correct/50)
    print("Specificity (Successfully predict cluster 2):",end=' ')
    print(predict_cluster2_correct/50)
    print((predict_cluster1_correct + predict_cluster2_correct) / (x.D.shape[0]))
    
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:48:39 2019

@author: Willy
"""
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import csv
from numpy.linalg import inv
from scipy.optimize import minimize

class Gaussian_Process(object):
    def __init__(self, X1, y1, X2, kernel_func, sigma_noise):
        self.X1 = X1
        self.y1 = y1
        self.X2 = X2
        self.kernel_func = kernel_func 
        self.sigma_noise = sigma_noise
        self.hypm        = [0.5,0.5,0,10]
    
    def rational_quadratic_hy(self,x):
        return x[0]**2*(1+ (1/x[1])*0.5*scipy.spatial.distance.cdist(self.X1, self.X1, 'sqeuclidean'))**(-x[1])
    
    def objective(self,x):
        cov = self.rational_quadratic_hy(x)
        value = -0.5 * self.y1.T.dot(inv(cov)).dot(self.y1)[0,0] - 0.5 * np.log(np.linalg.det(cov)) - 0.5 * self.y1.shape[0] * np.log(2*np.pi)
        return -value
    
    def optimiz_process(self):
        sol = minimize(self.objective,self.hypm, method='CG',options={'disp':True})
        self.hypm = sol.x
        print(sol.x)
        return
    
    def __call__(self):
        sigma11 = self.kernel_func(self.X1,self.X1,self.hypm) + self.sigma_noise * np.eye(X1.shape[0])
        sigma12 = self.kernel_func(self.X1,self.X2,self.hypm)
        solved  = scipy.linalg.solve(sigma11, sigma12, assume_a='pos').T
        mu2     = solved.dot(self.y1)
        sigma22 = self.kernel_func(self.X2,self.X2,self.hypm)
        sigma2  = sigma22 - solved.dot(sigma12)
        return mu2.flatten(),sigma2

def exponentiated_quadratic(xa, xb):
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    #print(np.exp(sq_norm))
    return np.exp(sq_norm)

def rational_quadratic(xa,xb,x):
    sq_norm = x[0]**2*(1 + (1/x[1])*0.5*scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean'))**(-x[1]) + x[2] + x[3]*xa.dot(xb.T)
    return sq_norm

def Read_Two_Colunm_File(file_name):
    with open(file_name,'r') as f_input:
        csv_input = csv.reader(f_input, delimiter=' ', skipinitialspace=True)
        x = []
        y = []
        for cols in csv_input:
            x.append(float(cols[0]))
            y.append(float(cols[1]))
    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    return x,y;

if __name__ == '__main__':
    # read .data file
    test_point = 1000
    domain = (-60, 60)
    X1, Y1 = Read_Two_Colunm_File('input.data')
    X2 = np.linspace(domain[0], domain[1], test_point).reshape(-1,1)
    GP = Gaussian_Process(X1, Y1, X2, rational_quadratic, 0.2)
    mu2,sigma2 = GP();
    Lsigma2 = np.sqrt(np.diag(sigma2))
    fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
    ax1.set_title('non-optimzed gaussian process');
    ax1.plot(X1, Y1, 'ko', linewidth=2, label='$(x_1, y_1)$')
    ax1.fill_between(X2.flat, mu2-2*Lsigma2, mu2+2*Lsigma2, color='red', alpha=0.15, label='$postior$')
    ax1.plot(X2, mu2, 'r-', lw=2, label='$\mu_{2|1}$') 
    ax1.legend()
    GP.optimiz_process()
    mu2,sigma2 = GP();
    Lsigma2 = np.sqrt(np.diag(sigma2))
    ax2.set_title('optimzed gaussian process');
    ax2.plot(X1, Y1, 'ko', linewidth=2, label='$(x_1, y_1)$')
    ax2.fill_between(X2.flat, mu2-2*Lsigma2, mu2+2*Lsigma2, color='red', alpha=0.15, label='$postior$')
    ax2.plot(X2, mu2, 'r-', lw=2, label='$\mu_{2|1}$')
    ax2.legend()
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 08:45:47 2019

@author: Willy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# define variable
n = 3
lamda = 10000
filename = "./data.txt"

def lu_crout(A):
    """
    """ 
    # initial LU matrix
    input_shape = A.shape
    n = input_shape[0]
    L = np.zeros(input_shape)
    U = np.eye(n)
    L[0,0] = A[0,0]
    
    for j in range(1,n):
        L[j,0] = A[j,0]
        U[0,j] = A[0,j] / L[0,0]
    
    for j in range(1,n-1):
        for i in range(j,n):
            L[i,j] = A[i,j]
            for k in range(0,j):
                L[i,j] = L[i,j] - L[i,k] * U[k,j]
        for k in range(j+1,n):
            U[j,k] = A[j,k]
            for i in range(0,j):
                U[j,k] = U[j,k] - L[j,i] * U[i,k]
            U[j,k] = U[j,k] / L[j,j]
            
    L[n-1,n-1] = A[n-1,n-1]
    for k in range(0,n-1):
        L[n-1,n-1] = L[n-1,n-1] - L[n-1,k] * U[k,n-1]
    return L,U

def inverse(A):
    """
    """
    input_shape = A.shape
    n = input_shape[0]
    b_matrix = np.eye(n)
    L,U = lu_crout(A)
    A_inverse = np.zeros(input_shape)
    
    for col in range(0,n):
        b_sub_matrix = b_matrix[:,col]
        d_matrix = np.zeros(b_sub_matrix.shape)
        d_matrix[0] = b_sub_matrix[0] / L[0,0]
        for i in range(1,n):
            d_matrix[i] = b_sub_matrix[i]
            for j in range(0,i):
                d_matrix[i] = d_matrix[i] - L[i][j] * d_matrix[j]
            d_matrix[i] = d_matrix[i] / L[i,i]
        A_inverse[n-1,col] = d_matrix[n-1]
        for i in range(n-2,-1,-1):
            A_inverse[i,col] = d_matrix[i]
            for j in range(i+1,n):
                A_inverse[i,col] = A_inverse[i,col] - U[i,j] * A_inverse[j,col]
    return A_inverse

def transpose(A): 
    """
    """
    input_shape = A.shape
    A_transpose = np.zeros((input_shape[1],input_shape[0]))
    for i in range(0,input_shape[1]):
        for j in range(0,input_shape[0]):
            A_transpose[i][j] = A[j][i]
    return A_transpose

def create_matrix(dataframe):
    """
    """
    global n
    n_n = n - 1
    A = np.zeros((dataframe.shape[0],n_n+1))
    for index, row in dataframe.iterrows():
        for i in range(n_n,-1,-1):
            value = 1.0
            for j in range(0,i):
                value = value * np.float64(row['X'])
            A[index,n_n-i] = value
    return A

def compute_total_error(matrix_a,matrix_b):
    return np.sum(np.square(np.subtract(matrix_a, matrix_b)))

def display_ploynomail(matrix_a):
    line = ""
    for i in range(0,matrix_a.shape[0]):
        if i == 0:
            line = " " + str(matrix_a[matrix_a.shape[0]-i-1])
        else:
            line = str(matrix_a[matrix_a.shape[0]-i-1]) + "x^" + str(i) + " + " + line
    print("Fitting Line: " + line)

def f(t1, ploynomial_parameter):
    global n
    y_ans = np.zeros(t1.shape[0])
    A     = np.ones((t1.shape[0],n))
    for i in range(0,A.shape[0]):
        x = 1.0
        for j in range(A.shape[1]-1,-1,-1):
            y_ans[i] = y_ans[i] + A[i,j] * x * ploynomial_parameter[j]
            x = x * t1[i]
    
    return y_ans
    
def main():
    global lamda
    global filename
    
    # LSE
    print("LSE:")
    # read txt file
    dataframe = pd.read_csv(filename , names=['X','Y'], dtype={'X':np.float64,'Y': np.float64})
    b = dataframe['Y'].values
    max_x_value = np.amax(dataframe['X'].values)
    min_x_value = np.amin(dataframe['X'].values)
    t1 = np.arange(min_x_value-1, max_x_value+1 , 0.1)
    # create a matrix
    A = create_matrix(dataframe)
    AAT = np.dot(transpose(A),A)
    AAT = AAT + lamda * np.eye(AAT.shape[0])
    AAT_inverse = inverse(AAT)
    tmp = np.dot(AAT_inverse,transpose(A))
    ploynomial_parameter = np.dot(tmp,b)
    compute_ans = np.dot(A,ploynomial_parameter)
    display_ploynomail(ploynomial_parameter)
    print("Total error: " + str(compute_total_error(compute_ans,b)))    
    print("")
    
    #Newtown's method
    print("Netwon's Method:")
    n_dataframe = pd.read_csv(filename , names=['X','Y'], dtype={'X':np.float64,'Y': np.float64})
    n_b = n_dataframe['Y'].values
    # create a matrix
    n_A = create_matrix(n_dataframe)
    n_AAT = np.dot(transpose(n_A),n_A)
    n_AAT = n_AAT
    n_AAT_inverse = inverse(n_AAT)
    tmp = np.dot(n_AAT_inverse,transpose(n_A))
    n_ploynomial_parameter = np.dot(tmp,n_b)
    n_compute_ans = np.dot(n_A,n_ploynomial_parameter)
    display_ploynomail(n_ploynomial_parameter)
    print("Total error: " + str(compute_total_error(n_compute_ans,n_b)))

    #plot
    plt.figure(figsize=(9, 6))
    plt.subplot(221)
    plt.plot(t1, f(t1, ploynomial_parameter), 'k-')
    plt.scatter(dataframe['X'].values,dataframe['Y'].values)
    
    plt.subplot(222)
    plt.plot(t1, f(t1, n_ploynomial_parameter), 'k-')
    plt.scatter(n_dataframe['X'].values,n_dataframe['Y'].values)
    
    plt.show()
    
    
if __name__ == '__main__':
    main()



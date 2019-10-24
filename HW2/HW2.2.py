# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 22:58:10 2019

@author: Willy
"""
INPUT_FILE = "./testfile.txt"

def compute_factorial(n,m):
    sum_value = 1.0
    positive_prob = m/n
    nagative_prob = 1 - positive_prob
    for i in range(n,m,-1):
        sum_value = sum_value * i
    for i in range(n-m,0,-1):
        sum_value = sum_value / i
    sum_value = sum_value* (positive_prob**m) * (nagative_prob**(n-m))
    return sum_value

def beta_function(prob,a,b):
    sum_value = 1.0
    sum_value = sum_value * (prob**(a-1)) * ((1-prob)**(b-1))
    for i in range(a+b,0,-1):
        sum_value = sum_value * i
    for i in range(a,0,-1):
        sum_value = sum_value / i
    for i in range(b,0,-1):
        sum_value = sum_value / i
    return sum_value

def main():
    f = open(INPUT_FILE)
    lines = f.readlines()
    a_value = 10
    b_value = 1
    for i in range(len(lines)):
        one_num = lines[i].count('1')
        zero_num= lines[i].count('0')
        all_num = one_num + zero_num
        print('case ',i+1,': '+lines[i][:-1])
        print('likelihood:',compute_factorial(all_num,one_num))
        print('Beta prior: a=',a_value,'b=',b_value)
        a_value += one_num
        b_value += zero_num
        print('Beta posterior: a=',a_value,'b=',b_value)
        print()
if __name__ == "__main__":
    main()
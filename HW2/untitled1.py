# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:40:00 2019

@author: Willy
"""

import numpy as np
img_array        = np.asarray([1,2,0,1,1,2])
unique, counts = np.unique(img_array, return_counts=True)
dic = dict(zip(unique, counts))
print(dic[1])

label_array      = np.zeros((60000,1))
unique, counts = np.unique(label_array, return_counts=True)
dic = dict(zip(unique, counts))
print(dic[0])
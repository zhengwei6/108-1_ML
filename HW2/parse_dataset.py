# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 08:36:27 2019

@author: Willy
"""
import numpy as np

def parse_datasets(img_path,label_path):
    img_magic_num    = 0
    label_magic_num  = 0
    img_array        = np.zeros((60000,28,28))
    label_array      = np.zeros((60000,1))
    # parse img path
    with open(img_path, "rb") as f:
        # read img_magic_num
        img_magic_num = int.from_bytes(f.read(4), "big")
        img_img_num   = int.from_bytes(f.read(4), "big")
        img_row_num   = int.from_bytes(f.read(4), "big")
        img_col_num   = int.from_bytes(f.read(4), "big")
        
        if img_magic_num != 2051:
            raise Exception("Wrong img path")
        
        for i in range(img_img_num):
            for j in range(img_row_num):
                for k in range(img_col_num):
                    img_array[i,j,k] = int.from_bytes(f.read(1), "big")
    
    # parse label path
    with open(label_path,"rb") as f:
        # read label_magic_num
        label_magic_num = int.from_bytes(f.read(4), "big")
        label_item_num  = int.from_bytes(f.read(4), "big")
        
        if label_magic_num != 2049:
            raise Exception("Wrong label path")
        
        for i in range(label_item_num):
            label_array[i,0] = int.from_bytes(f.read(1), "big")
    return img_array,label_array


'''  
img_array,label_array = parse_datasets("train-images.idx3-ubyte","train-labels.idx1-ubyte")
print(img_array.shape)
print(label_array.shape)
print(type(label_array[59999]))
print(type(img_array[59999,10,10]))
plt.imshow(img_array[59999])
'''
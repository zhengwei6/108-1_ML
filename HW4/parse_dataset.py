# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 08:36:27 2019

@author: Willy
"""
import numpy as np
from matplotlib import pyplot as plt

def parse_datasets(img_path,label_path,mode=0):
    img_magic_num    = 0
    label_magic_num  = 0

    imgs        = 0
    lbls      = 0

        
    # parse img path
    with open(img_path, "rb") as f:
        # read img_magic_num
        img_magic_num = int.from_bytes(f.read(4), "big")
        img_img_num   = int.from_bytes(f.read(4), "big")
        img_row_num   = int.from_bytes(f.read(4), "big")
        img_col_num   = int.from_bytes(f.read(4), "big")
        
        
        
        if img_magic_num != 2051:
            raise Exception("Wrong img path")
        imgs_elts = f.read(img_img_num * img_row_num * img_col_num)
        f.close()
        imgs = np.frombuffer(imgs_elts, dtype=np.uint8) 
        imgs = np.reshape(imgs, (28, 28, img_img_num), order='F') 
        imgs = np.transpose(imgs, (1,0,2))
        imgs = np.asarray(imgs , dtype=np.float) 
        
    # parse label path
    with open(label_path,"rb") as f:
        # read label_magic_num
        label_magic_num = int.from_bytes(f.read(4), "big")
        label_item_num  = int.from_bytes(f.read(4), "big")
        
        if label_magic_num != 2049:
            raise Exception("Wrong label path")
        label_elts = f.read(label_item_num)
        f.close()
        lbls = np.frombuffer(label_elts, dtype=np.uint8) 
        lbls = np.asarray(lbls, dtype=np.float)
        
    return imgs,lbls

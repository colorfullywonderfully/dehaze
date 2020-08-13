# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:48:11 2020

@author: 11574
"""
import os

dict_temp = {}


file = open('dict.txt','r')

def read_dict():

    for line in file.readlines():
        line = line.strip()
        k = line.split(':')[0]
        v = line.split(':')[1]
        dict_temp[k] = v
    
    file.close()
    
    for key,value in dict_temp.items():
        if key == 'input_path':
            input_path = value
        elif key == 'out_path':
            out_path = value
        elif key == 'net_path':
            net_path = value
        elif key == 'size':
            size = value

    return input_path, out_path, net_path, size
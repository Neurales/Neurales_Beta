# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 17:42:15 2020

@author: XZ-WAVE
"""
import numpy as np
import os 

files = os.listdir(os.getcwd()) #lists files in current working directory 

def get_files(path,exts=['.png','.jpg','.py','.zip']):
    files = os.listdir(path)
    num_exts = len(exts)
    all_files = []
    for n in range(num_exts):
        ext_files = {'{}_files'.format(exts[n]): []}
        for f in range(len(files)):
            if exts[n] in files[f]: #if extension at index n is found in files list at index f THEN DO BELOW
#                import pdb
#                pdb.set_trace()
                all_files.append(ext_files['{}_files'.format(exts[n])].append(files[f])) #append specific file at index f
           
    return {"files": ext_files}
        

def loop_three_times():
    for i in range(3):
        a=2
        b=4
        c=a+b
        print("c: {}".format(c))
        for j in range(2):
            d = 2*c
            print("d: {}".format(d))
            for k in range(2):
                e = d+c
                print("e: {}".format(e))
    return 'total: {}'.format(3*2*2)

def double_loop(num_lists=10,num_ints=50):
    data = []
    for i in range(num_lists): #for however many lists I have ....
        data.append('blah') #append string 'blah' 
        for j in range(num_ints): #for however many num_ints ...
            data.append(j+1) #append that integer 
    return data
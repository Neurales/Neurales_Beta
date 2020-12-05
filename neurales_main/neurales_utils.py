# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 00:23:32 2020

@author: XZ-WAVE
"""
import torch
import torch.nn as nn
import numpy as np
import random
import os
import pandas as pd
import sqlite3
import abc
import json 

'''
documentation for each function will come
'''


out2json= lambda x: json.dumps(x.tolist()) #dumps results (from class methods) to a json string
model2json= lambda x: json.dumps(str(x)) #dumps sklearn model to json string
model_name= lambda x: type(x).__name__ #gets the name of a class


 
def sql_dataset(sql_path,filename):
    conn=sqlite3.connect(sql_path)
    
    f = open(filename +'.csv', 'w')
    cursor = conn.cursor()
    cursor.execute('select * from database')
    while True:
        df = pd.DataFrame(cursor.fetchmany(1000))

        if len(df) == 0:
            break
        else:
            df.to_csv(f, header=False)
    f.close()
    cursor.close()
    conn.close()

choices=['red', 'green','blue','purple','yellow','magenta','cyan','orange','teal','azure']
def clf_colors(L,color_choice=choices):
    
    colors=np.random.choice(color_choice,size=len(L),replace=False)
    return tuple(colors)

    

        
    
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 21:31:05 2021

@author: xxx
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def get_comparison_data(vel):
    d_in=pd.read_excel("../shared/MD30/200_"+vel+"_in.xlsx", header=None)
    d_in=np.array(d_in.transpose().values.tolist())#[0]
    d_out=pd.read_excel("../shared/MD30/200_"+vel+"_out.xlsx", header=None)
    d_out=np.array(d_out.transpose().values.tolist()[0])
    return d_in, d_out;


vel="1.5"
avg=[]
temp2=[1,2,3,4,5,6]
for i in range(6):
    temp=pd.read_csv("../shared/MD30/250steps/"+vel+"vel/"+str(i+1)+"/writer_before.csv", header=None, sep=";")
    temp=temp[temp[0]==200]
    for j in range(4,10):
        temp2[j-4]=np.average(np.true_divide(np.array(temp[temp[6]==j].transpose().iloc[14].values.tolist()),np.array(temp[temp[6]==j].transpose().iloc[8].values.tolist())))
    avg.append(temp2.copy())

d_in, d_out=get_comparison_data(vel)

for i in range(6):
    a=0
    for element in avg:
       a+=element[i]
    
    print("a\n")
    plt.plot([i*36+18],[a/6],'go')
plt.plot(d_out)
plt.show()












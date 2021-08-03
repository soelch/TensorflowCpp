# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:53:24 2021

@author: xxx
"""

import numpy as np
from matplotlib import pyplot as plt

def sqr(x):
    return x*x*x-x*x

def dev(x):
    return 3*x*x-2*x

lr=0.1
curr=0.9
lstx=[curr]
lsty=[sqr(curr)]
lsterr=[sqr(curr)+4/27]
for i in range(15):
    curr=curr-(dev(curr))*lr
    lstx.append(curr)
    lsty.append(sqr(curr))
    lsterr.append(sqr(curr)+4/27)
    print(curr)
plt.plot(lstx,lsty)
plt.show()
plt.plot(lsterr)
plt.show()
res=np.vstack((np.array(lstx),np.array(lsty)))

np.savetxt("sgd_"+str(lr)+".csv",np.transpose(res),header="x,y",comments='',delimiter=",", fmt='%f')










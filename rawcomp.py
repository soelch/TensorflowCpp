# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 12:50:45 2021

@author: xxx
"""

import pandas as pd
import numpy as np
from numpy.random import randint
import math
from matplotlib import pyplot as plt

pd.set_option("display.precision", 12)

def couetteAnalytical(y, height, maxVelocity, time, accuracy, viscosity):
    approximation=0
    size=len(y)
    for i in range(1, accuracy):
        approximation+=(math.exp(-i*i*math.pi*math.pi*viscosity*time/(height*height))/
                        i)*np.sin(i*math.pi*(np.full((size,),1)-y/height))
    return (maxVelocity*y/height-
            (2*maxVelocity/math.pi)*approximation)
            
        
h=50
spaceStepSize=2.5
nSpaceSteps=12
lower=50-3.75
upper=50-31.25
maxVel=0.5
acc=100
vis=2.64
t=950
time=-0.125+0.25*t
spaceSteps=np.arange(upper, lower+1,spaceStepSize)
spaceStepscomp=np.arange(0, 50.1,0.1)
a=[]
b=[]

for t in range(300,1001,50):
    a=[]
    b=[]
    a=(couetteAnalytical(spaceSteps, h, maxVel, time, acc, vis))
    c=1
    for i in range(1,len(a)+1):
        if(i>3 and i<10):
            for j in range(36):
                b.append([c,a[-i]])
                c+=1
    print(b)
    b=np.around(np.array(b), decimals=5)
    
    np.savetxt("../shared/MD30/"+str(t)+"_"+str(maxVel)+"_rawcomp.csv", b,header="x,y",comments='',delimiter=",", fmt='%f')
    #np.savetxt("../shared/MD30/analytical/"+str(maxVel)+"/"+str(t)+"_comp.csv", np.transpose(np.vstack((np.arange(0, 50.1,0.1),compReal))),header="x,y",comments='',delimiter=",", fmt='%f')
    #load these with a=np.genfromtxt("..shared/MD30...")

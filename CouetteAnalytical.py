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
maxVel=1.0
acc=100
vis=2.64
t=1500
time=-0.125+0.25*t
spaceSteps=np.arange(upper, lower+1,spaceStepSize)
spaceStepscomp=np.arange(0, 50.1,0.1)
a=[]
b=[]

a=(couetteAnalytical(spaceSteps, h, maxVel, time, acc, vis))
for i in range(1,len(a)+1):
    if(i<4 or i>9):
        for j in range(144):
            b.append(a[-i])
    else:
        for j in range(144-36):
            b.append(a[-i])
b=np.around(np.array(b), decimals=5)
comp=couetteAnalytical(spaceStepscomp, h, maxVel, time, acc, vis)
compReal=[]
for i in range(1,len(comp)+1):
    compReal.append(comp[-i])
d_in=pd.read_excel("../shared/MD30/200_1.5_in.xlsx", header=None)
d_in=np.array(d_in.transpose().values.tolist())[0]
print(b-d_in)
plt.plot(b-d_in)
plt.show()
print(np.max(b-d_in))
d_filtered=[]
ctr=1
for i in range(12):
    d_filtered.append(d_in[ctr])
    ctr+=144
    if(i>2 and i<9):
        ctr-=36
plt.plot(np.arange(0, 50.1,0.1),compReal)
plt.plot(np.arange(3.75, 31.25+1,spaceStepSize),d_filtered,"go")
plt.show()

np.savetxt("../shared/MD30/analytical/"+str(maxVel)+"/"+str(t)+"_in.csv", b)
np.savetxt("../shared/MD30/analytical/"+str(maxVel)+"/"+str(t)+"_comp.csv", compReal)
#load these with a=np.genfromtxt("..shared/MD30...")
print(a)








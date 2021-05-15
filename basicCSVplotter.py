import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

df = pd.read_csv('output.csv', header=None)
fig, ax = plt.subplots()
print(df.values.tolist())
ax.plot(df.values.tolist()[0])
#ax.plot(np.linspace(0, df.size, num=df.size), df, "-", color=b)

plt.savefig("test.png")

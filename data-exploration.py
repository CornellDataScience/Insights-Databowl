#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.scatter(np.arange(0,10), np.arange(10,0, -1))
plt.show()

# %%
data = pd.read_csv(r'data/train.csv')
data_top = data.head() 
data_top

# %%

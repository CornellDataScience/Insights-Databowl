#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
data = pd.read_csv(r'data/train.csv')
data_top = data.head() 
data_top
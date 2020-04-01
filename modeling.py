# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# %%
data = pd.read_csv("../data/fe_data.csv", index_col=[0,1])
plays = data.loc[data.IsRusher]

# %%

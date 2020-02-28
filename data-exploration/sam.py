#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../data/clean_data.csv", index_col=[1,2])
data.head()

#%%
print('\n'.join(data.columns))

# %% [markdown]
# Each play has only one rusher, so we can use that to build a dataframe with
# one row for each play.

# %%
plays = data.loc[data.IsRusher,:]
plays.head()

# %%
plt.scatter(plays.Distance,plays.Yards)
plt.show()

#%% [markdown]
# How many yards did players gain?

#%%
sns.kdeplot(plays.Yards, shade=True)
plt.xlim((-10,25))
plt.title("Yards gained per play")
plt.show()

#%% [markdown]
# Do they usually gain yardage?

#%%
bars = np.array([data.Yards < 0, data.Yards == 0, data.Yards > 0])
bars = bars.sum(axis=1) / len(data.Yards)
plt.bar(['loss', '0 yards', 'gain'], bars)
plt.ylabel("Proportion of Plays")
plt.show()

# %% [markdown]
# We can actually use matplotlib to build a pretty useful visualization of the
# field during the play.

#%%
def rotate_points(arr, deg):
    theta = np.deg2rad(deg)

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    
    return np.squeeze((R @ (arr.T)).T)

cmap = {True:"red", False: "blue"}

play = data.loc[(2017090700,20170907000118), ["Team","X","Y","IsRusher","Orientation","YardsLeft","Distance"]]
m = np.array([(-1,0),(1,0),(0,3)])

plt.figure(figsize=(20,10))
row = play.loc[play.IsRusher,:].iloc[0,:]
plt.axvline(x=row.YardsLeft + 10, ls="--", c='k')
plt.axvline(x=row.YardsLeft+10+row.Distance, ls="--",c='y')

plt.ylim((0,53.333))
plt.axes().set_aspect("equal")

for i,row in play.iterrows():
    color = cmap[row.Team]
    if row.IsRusher:
        color = 'm'
    plt.scatter(row.X, row.Y, c=color, marker=rotate_points(m,row.Orientation), s=1000)

plt.show()

# %%

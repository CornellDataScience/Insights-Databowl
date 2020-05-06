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
    """ Rotate an array of points around the origin """
    theta = np.deg2rad(-deg)

    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    
    return np.squeeze((R @ (arr.T)).T)

cmap = {True:"red", False: "blue"}
m = np.array([(-1,-1),(1,-1),(0,2)])

def view_play(df):
    fig, ax = plt.subplots(figsize=(20,10))
    row = play.loc[play.IsRusher,:].iloc[0,:]

    scrim_x = row.YardLine
    if row.Field_eq_Possession:
        scrim_x = row.YardLine + 10
    else:
        scrim_x = 100 - row.YardLine + 10

    ax.axvline(x=scrim_x, ls="--", c='k')
    ax.axvline(x=scrim_x+row.Distance, ls="--",c='y')

    ax.set_ylim((0,53.333))
    ax.set_aspect("equal")

    for i,row in play.iterrows():
        color = cmap[row.Team]
        if row.IsRusher:
            color = 'm'
        ax.scatter(row.X, row.Y, c=color, marker=rotate_points(m,row.Orientation), s=500)
    
    return fig

play_id = np.random.choice(plays.index)
play = data.loc[play_id, :]
print(play_id)
view_play(play).show()

# %% [markdown]
# Note that while YardLine, YardsLeft, and most other metrics are standardized,
# the X and Y position data is not, so we have to figure out how to make sure
# that the players are on the correct side of the field etc.

# %%

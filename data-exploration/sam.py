#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_csv("../data/clean_data.csv", index_col=[1,2])
data.head()

#%%
print('\n'.join(data.columns))

# %% [markdown]
# We would really like to be able to quickly isolate individual plays

# %%
game_id = 2017090700
play_id = 20170907000118

play = data.loc[(game_id,play_id),:]

# %% [markdown]
# We can also do this with a `groupby`

# %%
plays = data.groupby(by=["GameId","PlayId"]).first()
plt.scatter(plays.Distance,plays.Yards)
plt.show()

#%%
drop_cols = ['X','Y','S','A','JerseyNumber','DisplayName','PlayerHeight','PlayerWeight','PlayerBirthDate','Position','PlayerCollegeName','NflId']
plays = data
#plays = data.drop(drop_cols, axis=1)
play_ids = set(data.PlayId)
ids = plays.loc[plays.Position == "QB","PlayId"]
print(len(ids))
print(len(play_ids))

#%% [markdown]
# How many yards did players gain?

#%%
sns.kdeplot(data.Yards, shade=True)
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


#%%
plt.scatter(data.Distance, data.Yards)
plt.show()

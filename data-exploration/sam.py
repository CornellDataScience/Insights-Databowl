#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

data = pd.read_csv("../data/train.csv", index_col=[0,1,2])
data.head()

#%%
print('\n'.join(data.columns))

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

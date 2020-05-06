# %% [markdown]
# # Feature Engineering
# Build some useful & predictive features and eliminate features we can't use.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

data = pd.read_csv("../data/clean_data.csv", index_col=[1,2])
plays = data.loc[data.IsRusher,:]

# %% [markdown]
# The data cleaning script doesn't actually provide whether each player is on
# the offensive or defensive side of the ball... fortunately it gives us the
# tools to find that without too much trouble.

# %%
data['IsOffense'] = data.HomePossession == data.Team

# %% [markdown]
# We also don't have a field that consistently finds the distance from the
# offense to the goal. We can engineer that too, using `Field_eq_Possession`
# which *seems* to tell us whether the offense is on their own side of the
# field, allowing us to transform `YardLine` appropriately.

# %%
# Efficient adjusted yardline field
data['YardLine_adj'] = data.YardLine
updated = 100 - data.loc[~data.Field_eq_Possession, 'YardLine']
data.loc[~data.Field_eq_Possession, 'YardLine_adj'] = updated

# %% [markdown]
# One might be tempted to assume that this is what `YardsLeft` indicates, but it
# actually doesn't. YardsLeft does not handle the reversal of the field well. We
# know that the `YardLine_adj` field is correct because I've cross-referenced it
# with the X/Y position data in another notebook.

# %%
plays = data.loc[data.IsRusher,:]

plt.scatter(plays.YardsLeft, plays.YardLine, c=plays.Field_eq_Possession)
plt.xlabel("YardsLeft")
plt.ylabel("YardLine")

plt.show()

# %% [markdown]
# Another issue we have is that the X position data is extremely noisy, since it
# most strongly correlates with the position of the line of scrimmage. By
# normalizing around the line of scrimmage, we can calculate instead each
# player's position relative to the line, which is likely much more predictive.

# %%
data["X_adj"] = data.X - (data.YardLine_adj + 10)
plt.scatter(data.loc[data.IsRusher, "X_adj"], data.loc[data.IsRusher, "Y"], alpha=0.1)
plt.show()

# %% [markdown]
# Both `S` and `Dis` seem to be representing the same thing--the speed of the
# player. Let's see how they relate.

# %%
plt.scatter(plays.Dis, plays.S)
plt.show()

# %% [markdown]
# There's some anomalies in the data here, but it looks like `S` is usually 10 *
# `Dis`. Let's make that substitution and drop the extraneous column.

# %%
data.S = data.Dis * 10
data.drop("Dis", axis=1, inplace=True)

# %% [markdown]
# We also have a lot of fields that we know are not going to be predictive.
# Let's drop them to produce a clean dataframe.

# %%
drop_cols = [
    "DisplayName",
    "JerseyNumber",
    "PossessionTeam",
    "FieldPosition",
    "OffensePersonnel",
    "DefensePersonnel",
    "PlayerCollegeName",
    "Position",
    "HomeTeamAbbr",
    "VisitorTeamAbbr",
    "Stadium",
    "Location",
    "StadiumType",
    "Turf",
    "GameWeather",
]

na_cols = [
    "Temperature",
    "Humidity",
    "WindSpeed",
    "WindDirection"
]

clean = data.drop(drop_cols + na_cols, axis=1)
clean = clean.dropna()

clean.head()

# %% [markdown]
# Now we can develop the detailed play data into one line per play

# %%
plays = clean.loc[clean.IsRusher,:]
plays = plays.drop(['IsRusher', 'IsOffense'], axis=1)

# %%
play = clean.loc[(2017090700,20170907000118)]

def clarity(df):
    df = df[['IsRusher','IsOffense','X','Y','S','A','Dis','Orientation','PlayerHeight','PlayerWeight','PlayerBMI']]
    rusher = df.loc[df['IsRusher'],:].iloc[0]
    
    

    return rusher

clarity(play)

# %% [markdown]
# We'll just dump this data to do predictions elsewhere.

# %%
clean.to_csv("../data/fe_data.csv")
plays.to_csv("../data/fe_plays_data.csv")


# %%

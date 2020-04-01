# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

plt.scatter(plays.YardsLeft, plays.YardLine_adj)
plt.xlabel("YardsLeft")
plt.ylabel("YardLine_adj")

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
# We also have a lot of fields that we know are not going to be predictive.
# Let's drop them to produce a clean dataframe.

# %%
drop_cols = [
    "X",
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
    "GameWeather",
]

clean = data.drop(drop_cols, axis=1)
plays = clean.loc[clean.IsRusher,:]

clean.head()

# %% [markdown]
# We'll just dump this data to do predictions elsewhere.

# %%
#clean.to_json("../data/fe_data.json", orient="index")

# %%
clean.to_csv("../data/fe_data.csv")

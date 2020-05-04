# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm

# %% [markdown]
# # What's a Convolution?
# 
# Gonna make a convolutional neural net

# %%
raw_data = pd.read_csv("data/fe_data.csv", index_col=[1])
raw_data.head(22)

# %%
print('\n'.join(raw_data.columns))

# %%
keep_columns = ["IsOffense","IsRusher","X","Y","S","A","Dir","Yards"]
data = raw_data[keep_columns]
data.head()

#%% [markdown]
# Filter out incomplete plays which don't have data for all players.

# %%
indices = data.index.unique()

wrong_shape = []
for i in tqdm(indices):
    if data.loc[i].shape != (22,len(keep_columns)):
        wrong_shape.append(i)

data = data.drop(wrong_shape)
data.shape

# %% [markdown]
# Let's engineer the components of each player's velocity. First we need to
# understand how to interpret the `Dir` variable.

# %%
sns.kdeplot(data.loc[data.IsOffense,'Dir'], shade=True, label='offense')
sns.kdeplot(data.loc[~data.IsOffense,'Dir'], shade=True, label='defense')
plt.legend()
plt.show()

# %% [markdown]
# Because the offense is usually pushing toward positive X, we can understand
# that $S_x = S \times sin(Dir)$

# %%
dir_rad = np.deg2rad(data.Dir)
S_x = data.S * np.sin(dir_rad)
S_y = -data.S * np.cos(dir_rad)

data['S_x'] = S_x
data['S_y'] = S_y

# %% [markdown]
# We can verify that we're correctly assigning the x speed by comparing offense
# and defense. Offense should generally have a positive x speed while defense
# should have a negative one.
# %%
sns.kdeplot(S_x.loc[data.IsOffense], label='offense x')
sns.kdeplot(S_x.loc[~data.IsOffense], label='defense x')
plt.legend()
plt.show()

# %% [markdown]
# We break down the dataframe into a list of dataframes, where each entry
# containa dataframe representing a play.

# %%
# Split data into play lists
cols = ['X', 'Y', 'S_x', 'S_y']

# Split offense and defense data
off_data = data.loc[data.IsOffense, cols]
def_data = data.loc[~data.IsOffense, cols]

split_plays = lambda df: [df.loc[i] for i in tqdm(df.index.unique())]

#plays = split_plays(data)
off_plays = split_plays(off_data)
def_plays = split_plays(def_data)

off_m = np.array([df.values for df in off_plays])
def_m = np.array([df.values for df in def_plays])

off_m.shape

# %% [markdown]
# ### Defender Velocity
#
# This layer consists of two features:
# 
# - defender $S_x$
# - defender $S_y$
#
# It does not include any information about the offense

# %%
# Defender velocity feature
def_vel = np.reshape(def_m[:,:,2:], (-1, 1, 11, 2)).repeat(11, axis=1)

# %%
# ### Offense-Defense Distances layer
#
# These features compare the (x,y) positions of the offensive and defensive
# players. This builds three features:
# 
# - x distance
# - y distance
# - euclidean distance
#
# This way the model is trained on both the absolute euclidean distance (so it
# doesn'have to calculate that) as well as each of its components, which
# preserve directionality as well.

# %%
# Euclidean distance feature
off_pos = np.reshape(off_m[:,:,:2], (-1, 11, 1, 2)).repeat(11, axis=2)
def_pos = np.reshape(def_m[:,:,:2], (-1, 1, 11, 2)).repeat(11, axis=1)

# Manhattan distance features (both x and y)
dist_xy = off_pos - def_pos

# Euclidean distance
dist_e = np.sqrt(np.square(dist_xy).sum(axis=3))

dist = np.concatenate([dist_xy, np.expand_dims(dist_e,3)], axis=3)
dist = dist.reshape(-1, 11, 11, 3)

# %% [markdown]
# ### Relative velocities
#
# These features compare the (x,y) velocity components of the offensive and
# defensive players. This builds two features:
# 
# - relative x velocity
# - relative y velocity
#
# The model may be able to convolute this with position to understand how
# players are moving in relation to each other.

# %%
# Relative velocity feature
off_vel = np.reshape(off_m[:,:,2:], (-1, 11, 1, 2)).repeat(11, axis=2)
def_vel = np.reshape(def_m[:,:,2:], (-1, 1, 11, 2)).repeat(11, axis=1)

d_vel = off_vel - def_vel
d_vel = d_vel.reshape(-1, 11, 11, 2)

# %%
target = data.loc[data.IsRusher, "Yards"]

def pdf(n):
    arr = [0] * 199
    arr[n+99] = 1
    return arr

features = [
    def_vel,
    dist,
    d_vel
]

x = np.concatenate(features, axis=3)
y = np.array(list(target.map(pdf)), dtype=np.float32)

print("X shape:", x.shape)
print("Y shape:", y.shape)

# %% [markdown]
# ## Build model
# We use three layers of 2D convolution

# %%
def compile_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(x.shape[1:], name='input'),
        tf.keras.layers.Conv2D(64, kernel_size=(1,1), strides=(1,1), activation='relu'),
        tf.keras.layers.Conv2D(64, kernel_size=(1,1), strides=(1,1), activation='relu'),
        tf.keras.layers.Conv2D(64, kernel_size=(1,1), strides=(1,1), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(1,10)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', name='d_relu'),
        tf.keras.layers.Dense(199, activation='softmax')
    ])

    mse = tf.keras.losses.MeanSquaredError()
    def pdf_CRPS(y_true_pdf, y_pred_pdf):
        y_true_cdf = tf.math.cumsum(y_true_pdf, axis=1)
        y_pred_cdf = tf.math.cumsum(y_pred_pdf, axis=1)

        return mse(y_true_cdf, y_pred_cdf)

    model.compile(optimizer='adam', loss=pdf_CRPS)
    return model

model = compile_model()
history = model.fit(x,y, epochs=5)

# %% [markdown]
# ## Training Report

# %%
hist = pd.DataFrame(history.history)
fig, axes = plt.subplots(1,2, figsize=(16,8))
baseline_mse = 0.013200
goal_mse = 0.012864
best_mse = 0.011658

for start,ax in zip([0,-10], axes):
    ax.plot(hist.loss[start:], label='loss')
    #ax.plot(hist.val_loss[start:], label='val_loss')

    ax.axhline(baseline_mse, label='baseline', c='red', ls='--')
    ax.axhline(goal_mse, label='goal', c='purple', ls='--')
    ax.axhline(best_mse, label='best', c='green', ls='--')
    
    ax.legend()
    ax.grid()

    ax.set_xlabel("Epochs")

axes[1].set_xticks(np.arange(start % hist.index.max(), hist.index.max(), 2))

print(hist.tail(5))
fig.show()

# %%

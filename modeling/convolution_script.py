import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras.backend as K
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("CONVOLUTING")

raw_data = pd.read_csv("data/fe_data.csv", index_col=[1])
raw_data.head()

keep_columns = ["IsOffense","IsRusher","X","Y","S","A","Dir","Yards"]
data = raw_data[keep_columns]
data.head()

indices = data.index.unique()

wrong_shape = []
for i in tqdm(indices):
    if data.loc[i].shape != (22,len(keep_columns)):
        wrong_shape.append(i)

data = data.drop(wrong_shape)
data.shape

sns.kdeplot(data.loc[data.IsOffense,'Dir'], shade=True, label='offense')
sns.kdeplot(data.loc[~data.IsOffense,'Dir'], shade=True, label='defense')
plt.legend()
#plt.show()

dir_rad = np.deg2rad(data.Dir)
S_x = data.S * np.sin(dir_rad)
S_y = -data.S * np.cos(dir_rad)

data['S_x'] = S_x
data['S_y'] = S_y

# ==================
# Data Reformatting

# Split data into play lists
cols = ['X', 'Y', 'S_x', 'S_y']

off_axis = 1 # the axis along which the offensive player changes
def_axis = 2 # the axis along which the defensive player changes

df = data.loc[:,cols]
#norm_data = (df - df.mean()) / df.std()
#norm_data.head()

# Split offense and defense data
off_data = df.loc[data.IsOffense & ~data.IsRusher]
def_data = df.loc[~data.IsOffense]
rus_data = df.loc[data.IsRusher]

split_plays = lambda df: [df.loc[i,:] for i in tqdm(df.index.unique())]

#plays = split_plays(data)
off_plays = split_plays(off_data)
def_plays = split_plays(def_data)
rus_plays = split_plays(rus_data)

off_m = np.array([df.values for df in off_plays])
def_m = np.array([df.values for df in def_plays])
rus_m = np.array([df.values for df in rus_plays])

# Expand and repeat arrays to match shapes
off_m = np.expand_dims(off_m, axis=def_axis).repeat(11, def_axis)
def_m = np.expand_dims(def_m, axis=off_axis).repeat(10, off_axis)

rus_m = np.expand_dims(rus_m, off_axis).repeat(10, off_axis)
rus_m = np.expand_dims(rus_m, def_axis).repeat(11, def_axis)

print("Offense data shape", off_m.shape)
print("Defense data shape", def_m.shape)
print("Rusher data shape", rus_m.shape)

# Feature Engineering

def_vel = def_m[:,:,:,:]
def_vel.shape

# Defense vs Offense
# X and Y components of relative position and velocity
off_def_rel_comp = off_m - def_m

# Euclidean distance
off_def_ed = np.sqrt(np.square(off_def_rel_comp[:,:,:,:2]).sum(axis=3))
off_def_ed = np.expand_dims(off_def_ed, 3)

off_def = np.concatenate([off_def_rel_comp, off_def_ed], axis=3)
off_def.shape

# Defender vs. Rusher
# Components speed & velocity
def_rus_rel_comp = def_m - rus_m

# Euclidean distance
def_rus_ed = np.sqrt(np.square(def_rus_rel_comp[:,:,:,:2]).sum(axis=3))
def_rus_ed = np.expand_dims(def_rus_ed, 3)

def_rus = np.concatenate([def_rus_rel_comp, def_rus_ed], axis=3)
def_rus.shape

target = data.loc[data.IsRusher, "Yards"]

def pdf(n):
    arr = [0] * 199
    arr[n+99] = 1
    return arr

features = [
    #off_m,
    def_m,
    off_def,
    def_rus
]

x = np.concatenate(features, axis=3)
y = np.array(list(target.map(pdf)), dtype=np.float32)

x_train, x_test, y_train, y_test = train_test_split(x,y)

print("X shape:", x.shape)
print("Y shape:", y.shape)

# %%
def compile_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(x.shape[1:], name='input'),

        tf.keras.layers.Conv2D(64, kernel_size=(1,1), strides=(1,1), activation='relu'),
        tf.keras.layers.Conv2D(64, kernel_size=(1,1), strides=(1,1), activation='relu'),
        tf.keras.layers.Conv2D(64, kernel_size=(1,1), strides=(1,1), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(1,10)),
        
        tf.keras.layers.Lambda(lambda y: K.squeeze(y,2)),

        tf.keras.layers.Conv1D(128, kernel_size=(1), strides=(1), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(160, kernel_size=(1), strides=(1), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(96, kernel_size=(1), strides=(1), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=(10)),

        tf.keras.layers.Lambda(lambda y: K.squeeze(y,1)),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(199, activation='softmax', name='output')
    ])

    mse = tf.keras.losses.MeanSquaredError()
    def pdf_CRPS(y_true_pdf, y_pred_pdf):
        y_true_cdf = tf.math.cumsum(y_true_pdf, axis=1)
        y_pred_cdf = tf.math.cumsum(y_pred_pdf, axis=1)

        return mse(y_true_cdf, y_pred_cdf)

    model.compile(optimizer='adam', loss=pdf_CRPS)
    return model

model = compile_model()

logdir="./logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

history = model.fit(x_train, y_train, 
    epochs=20,
    validation_split=0.2,
    shuffle=False,
    callbacks=[tensorboard_callback]
)

hist = pd.DataFrame(history.history)
fig, axes = plt.subplots(1,2, figsize=(16,8))
baseline_mse = 0.013200
goal_mse = 0.012864
best_mse = 0.011658

for start,ax in zip([0,-10], axes):
    ax.plot(hist.loss[start:], label='loss')
    ax.plot(hist.val_loss[start:], label='val_loss')

    ax.axhline(baseline_mse, label='baseline', c='red', ls='--')
    ax.axhline(goal_mse, label='goal', c='purple', ls='--')
    ax.axhline(best_mse, label='best', c='green', ls='--')
    
    ax.legend()
    ax.grid()

    ax.set_xlabel("Epochs")

axes[1].set_xticks(np.arange(start % hist.index.max(), hist.index.max(), 2))

print(hist.tail(5))
fig.savefig("training.png")
#fig.show()

model.evaluate(x_test, y_test)

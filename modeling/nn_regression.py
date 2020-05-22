# %% [markdown]
# # Applying deep learning models to Databowl
# Here I use tensorflow to build a feedforward neural net to predict yards
# gained on each rushing play.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_docs as tfdocs

np.set_printoptions(1)

# %% [markdown]
# The feature engineering notebok does a pretty good job of cleaning up the
# data, but leaves behind just a couple NA entries. We drop those rows, mostly
# for convenience.

# %%
data = pd.read_csv("data/fe_data.csv", index_col=[0,1])
data.dropna(inplace=True)

# %% [markdown]
# For this iteration, we're using only one line of data for each play--the data
# entry for the rusher. Because `IsRusher` and `IsOffense` are redundant as a
# result, we'll drop those columns from the the dataset. Then we can normalize
# all values so that they are well-interpreted by the model.

# %%
plays = data.loc[data.IsRusher]
plays = plays.astype(np.float32)

target = plays.pop("Yards")

plays.drop(["IsRusher","IsOffense"], axis=1, inplace=True)
plays_stats = plays.describe().transpose()

def norm(x):
    return (x - plays_stats['mean']) / plays_stats['std']

norm_data = norm(plays)

# %%
x = norm_data.values.astype(np.float32)
y = target.values.astype(np.float32)

dataset = tf.data.Dataset.from_tensor_slices((x, y))

# %% [markdown]
# We can verify that the data arrived in the `tf.data` API all in one piece.

# %%
for f,t in dataset.take(5):
    print(f"Features:\n{f}")
    print(f"Target: {t}")
    print()

# %% [markdown]
# Now we can segment this into a training dataset using the API.

# %%
train_data = dataset.shuffle(len(norm_data)).batch(100)

# %%
def compile_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(x.shape[1], name='input'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        #tf.keras.layers.Dropout(0.2),
        #tf.keras.layers.Dense(16, activation='relu', name='hidden'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='linear', name='output')
    ])

    rms = tf.keras.optimizers.RMSprop(0.001)
    sgd = tf.keras.optimizers.SGD()
    adam = tf.keras.optimizers.Adam()

    model.compile(optimizer=rms, loss='mse', metrics=['mse','mae'])
    return model

EPOCHS = 50
model = compile_model()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(x,y, 
    validation_split=0.2, 
    epochs=EPOCHS, 
    callbacks=[early_stop],
    verbose=0
)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
plt.plot(hist.loss, label='Training MSE')
plt.plot(hist.val_loss, label='Validation MSE')
plt.legend()
plt.show()

# %% [markdown]
# It's pretty clear that this model is immediately overfitting. We'll need to
# look into ways to fundamentally improve it or to drastically improve our
# feature engineering.

# %% [markdown]
# We can compare this to a baseline model that just predicts the mean value of
# yards for any inputs.

# %%
mse_data = ((y-y.mean())**2).mean()
mae_data = (np.abs(y-y.mean())).mean()
print("MSE of Yards against their mean:", mse_data)
print("MAE of Yards against their mean:", mae_data)

# %% [markdown]
# The MSE of this baseline model is 33.1 yds^2, about the same as our model,
# which indicates that our model has very little predictive power.

# %%
y_pred = model.predict(x)
plt.scatter(y, y_pred, alpha=0.05)
plt.xlabel("Actual Yards")
plt.ylabel("Predicted Yards")
plt.show()

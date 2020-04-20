# %% [markdown]
# # Modeling Notebook pt. 2
# This is a new notebook to work on modeling the data with neural nets,
# reflecting the specific loss function used by the Databowl challenge. Databowl
# uses a Cumulative Ranked Probability Score (CRPS) to evaluate models, which is
# given by:
#
# $$ 
# C = \frac{1}{199N} \sum^N_{m=1} \sum^{99}_{n=-99} (P(y \leq n) - H(n-Y_m))^2
# $$
#
# where $N$ is the number of samples, $P$ is the predicted probability
# distribution function being evaluated, $Y_m$ is the true number of yards the
# ball was carried on play $m$, and $H$ is the Heaviside step function which is
# 0 for negative inputs and 1 for positive inputs.
# 
# What does this evaluate? CRPS is the mean squared difference between two
# cumulative probability distributions: $P$ and $H$. Therefore, we can generate
# the target distribution and then just use MSE as our loss function on a
# regression with 199 outputs. This is a little rough, but the model performs
# relatively well.

# %%
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split

np.set_printoptions(1)

# %% [markdown]
# Feature engineering has already been done in another notebook, so all we need
# do is import the csv file we created. We index on both game id and play id, so
# we set both of those as the index.

# %%
data = pd.read_csv("data/fe_plays_data.csv", index_col=[0,1])
data.dropna(inplace=True)
data.head()

# %% [markdown] 
# We have to normalize all the data around its mean in order for the neural
# network to train effectively. We also drop a couple columns that don't contain
# any useful information.
#
# We define `cdf` to generate the "true" matrix that we will compare against in
# the predicted matrix. We convert all the target values (integer numbers of
# yards) into cumulative probability distributions with the method described by
# the CRPS implementation in the Databowl evaluation page.

# %%
data = data.astype(np.float32)
target = data.pop("Yards").astype(np.int)

data.drop(["IsRusher", "IsOffense"], axis=1, inplace=True)
stats = data.describe().transpose()

def norm(x):
    return (x - stats['mean']) / stats['std']

norm_data = norm(data)

def cdf(n):
    return [0]*(99+n) + [1]*(100-n)

target_m = np.array(list(target.map(cdf)))

# %% [markdown]
# We use sklearn to build our KFolds.

# %%
x = norm_data.values
y = target_m
x_train, x_test, y_train, y_test = train_test_split(norm_data.values, target_m)

# %% [markdown]
# This distribution-based model (with CRPS loss) is a bit harder to interpret,
# so we establish a baseline to evaluate against. We build this baseline by
# creating a cumulative probability distribution from the distribution of the
# target variable (yards gained on the play).

# %%
baseline = [(target < i).mean() for i in range(-99, 100)]

y_pred = np.array([baseline] * len(y))
baseline_mse = np.square(y_pred - y).mean()
print(f"Baseline MSE: {baseline_mse:.4f}")

plt.plot(list(range(-99,100)), baseline)
plt.title("Baseline model")
plt.xlabel("Yards")
plt.grid()

plt.show()

# %% [markdown]
# ## Train Model
# This is a relatively simple feed-forward neural network regression that
# produces 199 outputs. Because the CRPS metric is essentially
# mean-squared-error on a probability distribution, we are able to simply use
# the mse metric as long as we generate the target distributions beforehand.
#
# I would also at some point like to introduce K-fold cross validation in order
# evaluate the performance of the model more effectively.

# %%
def compile_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(x.shape[1], name='input'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(199)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

EPOCHS = 50
stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=.00005)
model = compile_model()
history = model.fit(x_train,y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    callbacks=[stop],
    verbose=0
)

# %% [markdown]
# ## Model training performance

# %%
hist = pd.DataFrame(history.history)
fig, ax = plt.subplots(1,2, figsize=(16,8))
goal_mse = 0.012864
best_mse = 0.011658

for start,ax_i in zip([0,-10], ax):
    ax_i.plot(hist.loss[start:], label='loss')
    ax_i.plot(hist.val_loss[start:], label='val_loss')

    ax_i.axhline(baseline_mse, label='baseline', c='red', ls='--')
    ax_i.axhline(goal_mse, label='goal', c='purple', ls='--')
    ax_i.axhline(best_mse, label='best', c='green', ls='--')
    
    ax_i.legend()
    ax_i.grid()

    ax_i.set_xlabel("Epochs")

ax[1].set_xticks(np.arange(start % hist.index.max(), hist.index.max(), 2))

print(hist.tail(5))
fig.show()

# %% [markdown]
#
# ## Evaluation
# Our primary concern is preventing overfitting, as we have many features and
# not a lot of significant information.
#
# We can examine the model's performance with a case study, comparing a long, 99
# yard run play

# %%
y_pred_long = model.predict(x[20127:20128])[0]
y_pred_short = model.predict(x[:1])[0]


plt.plot(list(range(-99,100)), y_pred_short, label='short')
plt.plot(list(range(-99,100)), y_pred_long, label='long')

plt.title("Prediction Distribution")
plt.xlabel("Yards")
plt.grid()
plt.legend()

plt.show()

# %% [markdown]
# The fact that the predictions for these two plays differ significantly shows
# thour model has some predictive power. Finally, we can check the model's
# performancen the test set that we built earlier.

# %%
score = model.evaluate(x_test, y_test)
print(f"Test MSE: {score:.4f}")

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# How many rows do we want to read?
#n = 5e6 # With 5e8, about LB 1.63

# Import
#float_data = pd.read_csv("input/train/train_batch_num%s.csv"%1, #nrows=n,
#                         dtype={"acoustic_data": np.float32, "time_to_failure": np.float32})

float_data = pd.read_csv("input/train.csv", #nrows=n,
                         dtype={"acoustic_data": np.float32, "time_to_failure": np.float32})

float_data = float_data.values

# Helper function for the data generator. Basically we want to extract mean, standard deviation, min and max per time step.
def extract_features(z):
     return np.c_[z.mean(axis=1),
                  np.median(np.abs(z), axis=1),
                  z.std(axis=1),
                  z.max(axis=1),
                  z.min(axis=1)]

# For a given ending position "last_index", we split the last 150'000 values of "x" into 150 pieces of length 1000 each.
# From each piece, 16 features are extracted. This results in a feature matrix of dimension (150 time steps x 16 features).
def create_X(x, last_index=None, n_steps=150, step_length=1000):
    if last_index == None:
        last_index=len(x)

    assert last_index - n_steps * step_length >= 0

    # Reshaping and approximate standardization with mean 5 and std 3.
    #[:]
    temp = (x[(last_index - n_steps * step_length):last_index].reshape(n_steps, -1) - 5 ) / 3

    # Extracts features of sequences of full length 1000, of the last 100 values and finally also
    # of the last 10 observations.
    return np.c_[extract_features(temp),
                 extract_features(temp[:, -step_length // 10:]),
                 extract_features(temp[:, -step_length // 100:]),
                 temp[:, -1:]]
'''
def gen_X(x, last_index=None, n_steps=150, step_length=1000):
    if last_index == None:
        last_index = 150000*4197 + pd.read_csv("input/train/train_batch_num4197.csv").shape[0]
    
    assert last_index - n_steps * step_length >= 0
'''



# We call "extract_features" three times, so the total number of features is 3 * 5 + 1 (last value) = 16
n_features = 16

# The generator randomly selects "batch_size" ending positions of sub-time series. For each ending position,
# the "time_to_failure" serves as target, while the features are created by the function "create_X".
def generator(data, min_index=0, max_index=None, batch_size=16, n_steps=150, step_length=1000):
    if max_index is None:
        max_index = len(data) - 1

    while True:
        # Pick indices of ending positions
        rows = np.random.randint(min_index + n_steps * step_length, max_index, size=batch_size)

        # Initialize feature matrices and targets
        samples = np.zeros((batch_size, n_steps, n_features))
        targets = np.zeros(batch_size, )

        for j, row in enumerate(rows):
            print(data.shape)
            samples[j] = create_X(data[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)
            targets[j] = data[row, 1]
        yield samples, targets

# Initialize generators
batch_size = 32

train_gen = generator(float_data, batch_size=batch_size)
valid_gen = generator(float_data, batch_size=batch_size)

#from tensorflow.contrib.compiler import xla
# Define model

from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU, RNN, SimpleRNNCell
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint, BaseLogger

cb = [ModelCheckpoint("model.hdf5", monitor='val_loss', save_weights_only=False, period=3)]

model = Sequential()
model.add(CuDNNGRU(48, input_shape=(None, n_features)))
#model.add(SimpleRNN(48, input_shape=(None, 16), batch_input_shape = (32, 150, 16)))
#model.add(Conv1D(48, kernel_size=(1),
#             activation='relu',
#             input_shape=(None, 16)))
#model.add(Dense(48, input_shape=(None, 16)))
#model.add(Dense(48, input_shape=(None, n_features)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# Compile and fit model
model.compile(optimizer=adam(lr=0.0005), loss="mae")

history = model.fit_generator(train_gen,
                              steps_per_epoch=1000,#n_train // batch_size,
                              epochs=30,
                              verbose=0,
                              callbacks=cb,
                              validation_data=valid_gen,
                              validation_steps=100)#n_valid // batch_size)

model.summary()
# Visualize accuracies
import matplotlib.pyplot as plt

def perf_plot(history, what = 'loss'):
    x = history.history[what]
    val_x = history.history['val_' + what]
    epochs = np.asarray(history.epoch) + 1

    plt.plot(epochs, x, 'bo', label = "Training " + what)
    plt.plot(epochs, val_x, 'b', label = "Validation " + what)
    plt.title("Training and validation " + what)
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
    return None

perf_plot(history)

# Load submission file
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

# Load each test data, create the feature matrix, get numeric prediction
for i, seg_id in enumerate(submission.index):
  #  print(i)
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    x = seg['acoustic_data'].values
    submission.time_to_failure[i] = model.predict(np.expand_dims(create_X(x), 0))

submission.head()

# Save
submission.to_csv('submission.csv')
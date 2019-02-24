import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import sys
import os

chunksize = 150000
float_data = pd.read_csv("input/train.csv", chunksize=chunksize,
                         dtype={"acoustic_data": np.float32, "time_to_failure": np.float32})

#float_data = float_data.values

def extract_features(z):
     return np.c_[z.mean(axis=1),
                  np.median(np.abs(z), axis=1),
                  z.std(axis=1),
                  z.max(axis=1),
                  z.min(axis=1)]


def show_X(data):
    for i, x in enumerate(data):
        print("chunk num: %s"%(i) + "\r")
        temp = (x-5)/3#hypotest?Z
        features = extract_features(temp)
        print("mean: %s"%features[0])
        print("median: %s"%features[1])
        print("std: %s"%features[2])
        print("max: %s"%features[3])
        print("min: %s"%features[4])
        time.sleep(2)

def create_X(x, last_index=None, n_steps=150, step_length=1000):
    if last_index == None:
        last_index=len(x)
    #print(last_index)
    #print(n_steps * step_length)
    assert last_index - n_steps * step_length >= 0

    # Reshaping and approximate standardization with mean 5 and std 3.
    #[:]
    #why 5, 3?
    temp = (x[(last_index - n_steps * step_length):last_index].reshape(n_steps, -1) - 5 ) / 3
    #temp = (x[(last_index - n_steps * step_length):(n_steps * step_length)].reshape(n_steps, -1) - 5 ) / 3
    # Extracts features of sequences of full length 1000, of the last 100 values and finally also
    # of the last 10 observations.
    return np.c_[extract_features(temp),
                 extract_features(temp[:, -step_length // 10:]),
                 extract_features(temp[:, -step_length // 100:]),
                 temp[:, -1:]]

n_features = 16

BATCH_LIMIT = 5 #up to 4196/2
#also try train_batch_num implementation
#since certain ranges not considered, limited by chunksize chunking?

def generator(file_name, batch_size=16, n_steps=150, step_length = 1000):
    epoch = 0
    while True:
        chunksize = 2*n_steps*step_length
        float_data = pd.read_csv("input/train.csv", chunksize=chunksize,
            dtype={"acoustic_data": np.float32, "time_to_failure": np.float32})
        for i, data in enumerate(float_data):
            if i == BATCH_LIMIT:
                break
            #if i == len(float_data):
            if data.shape[0] != chunksize:
                #idk end edge case
                epoch += 1
                continue
            data = data.values
            rows = np.random.randint(n_steps*step_length, chunksize, size=batch_size)#makes cv here?
            samples = np.zeros((batch_size, n_steps, n_features))
            targets = np.zeros(batch_size, )
            sample_ranges = None
            for j, row in enumerate(rows):
                #try:
                #print("row: %s"%row)
                samples[j] = create_X(data[:, 0], last_index=row, n_steps=n_steps, step_length=step_length)
                #except TypeError:
                #    print(data.shape)
                targets[j] = data[row, 1]
                sample_range = np.arange(i*chunksize+row-chunksize, i*chunksize+row)
                if sample_ranges is None:
                    sample_ranges = sample_range
                else:
                    sample_ranges = np.vstack(sample_ranges, sample_range)

            np.expand_dims(targets, 1)
            
            sample_range = np.arange(i*chunksize, i*chunksize)#or might be 2d with shape(chunksize, batch_size)
            yield samples, targets, sample_ranges, epoch

#show_X(float_data)

batch_size = 32

train_gen = generator(float_data, batch_size=batch_size)
valid_gen = generator(float_data, batch_size=batch_size)

for i, batch in enumerate(train_gen):
    print("IT IS I: %s"%i)
    print(batch[0].shape)
    print(batch[1].shape)
    break

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU, RNN, SimpleRNNCell, Flatten, SimpleRNN, Conv1D
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint, BaseLogger

NB_EPOCHS = 5       # number of times the model sees all the data during training

N_FORWARD = 8       # train the network to predict N in advance (traditionnally 1)
RESAMPLE_BY = 5     # averaging period in days (training on daily data is too much)
RNN_CELLSIZE = 32  # size of the RNN cells
N_LAYERS = 2        # number of stacked RNN cells (needed for tensor shapes but code must be changed manually)
SEQLEN = 32        # unrolled sequence length
BATCHSIZE = 32      # mini-batch size
DROPOUT_PKEEP = 0.7 # probability of neurons not being dropped (should be between 0.5 and 1)
ACTIVATION = tf.nn.tanh # Activation function for GRU cells (tf.nn.relu or tf.nn.tanh)

JOB_DIR  = "checkpoints"

def model_rnn_fn(features, Hin, labels, step, dropout_pkeep):
    X = features  # shape [BATCHSIZE, SEQLEN, 2], 2 for (agg_features)
    batchsize = tf.shape(X)[0]
    seqlen = tf.shape(X)[1]
    pairlen = tf.shape(X)[2] # should be 2 (acoustic_data, time_to_failure)?
    
    cells = [tf.nn.rnn_cell.GRUCell(RNN_CELLSIZE, activation=ACTIVATION) for _ in range(N_LAYERS)]
    # dropout useful between cell layers only: no output dropout on last cell
    cells = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = dropout_pkeep) for cell in cells]
    # a stacked RNN cell still works like an RNN cell
    cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=False)
    # X[BATCHSIZE, SEQLEN, 2], Hin[BATCHSIZE, RNN_CELLSIZE*N_LAYERS]
    # the sequence unrolling happens here
    Yn, H = tf.nn.dynamic_rnn(cell, X, initial_state=Hin, dtype=tf.float32)
    # Yn[BATCHSIZE, SEQLEN, RNN_CELLSIZE]
    Yn = tf.reshape(Yn, [batchsize*seqlen, RNN_CELLSIZE])
    Yr = tf.layers.dense(Yn, 32) # Yr [BATCHSIZE*SEQLEN, 2]
    Yr = tf.reshape(Yr, [batchsize, seqlen, 32]) # Yr [BATCHSIZE, SEQLEN, 2]
    Yout = Yr[:,-N_FORWARD:,:] # Last N_FORWARD outputs Yout [BATCHSIZE, N_FORWARD, 2]
    
    loss = tf.losses.mean_squared_error(Yr, tf.reshape(labels, [1, 1, 32])) # labels[BATCHSIZE, SEQLEN, 2]
    
    lr = 0.001 + tf.train.exponential_decay(0.01, step, 1000, 0.5)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)
    
    return Yout, H, loss, train_op, Yr

tf.reset_default_graph() # restart model graph from scratch

# placeholder for inputs
#with tf.device("/device:GPU:0"):
#(32, 150, 16)-samples.shape -> fed into features? transpose??
#(32,)-targets.shape -> fed into labels?
Hin = tf.placeholder(tf.float32, [None, RNN_CELLSIZE * N_LAYERS], name="Hin")
features = tf.placeholder(tf.float32, [None, None, 16], name="features") # [BATCHSIZE, SEQLEN, 2]
labels = tf.placeholder(tf.float32, [32], name="labels") # [BATCHSIZE, SEQLEN, 2]??
step = tf.placeholder(tf.int32, name="step")
dropout_pkeep = tf.placeholder(tf.float32, name="dropout_pkeep")

# instantiate the model
Yout, H, loss, train_op, Yr = model_rnn_fn(features, Hin, labels, step, dropout_pkeep)

# variable initialization
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run([init])
saver = tf.train.Saver(max_to_keep=1)

losses = []
indices = []
last_epoch = 99999

TRAIN_EPOCHS = 1000
'''
for i, (next_features, next_labels, dates, epoch, fileid) in enumerate(
    
    utils_batching.rnn_multistation_sampling_temperature_sequencer(train_filenames,
                                                                   RESAMPLE_BY,
                                                                   BATCHSIZE,
                                                                   SEQLEN,
                                                                   N_FORWARD,
                                                                   NB_EPOCHS, tminmax=True)):
'''
for i, (next_features, next_labels, evalranges, epoch) in enumerate(train_gen):
    if epoch == TRAIN_EPOCHS:
        print(f"Epoch {epoch} reached, saving model...")
        break
    if epoch != last_epoch:
        batchsize = next_features.shape[0]
        H_ = np.zeros([batchsize, RNN_CELLSIZE * N_LAYERS])
        print("State reset")
    #train
    feed = {Hin: H_, features: next_features, labels: next_labels, step: i, dropout_pkeep: DROPOUT_PKEEP}
    Yout_, H_, loss_, _, Yr_ = sess.run([Yout, H, loss, train_op, Yr], feed_dict=feed)
    
    # print progress
    if i%20 == 0:
        print("{}: epoch {} loss = {}".format(i, epoch, np.mean(loss_)))
        sys.stdout.flush()
    if i%10 == 0:
        losses.append(np.mean(loss_))
        indices.append(i)
        
    last_epoch = epoch

SAVEDMODEL = JOB_DIR + "/ckpt" + str(int(time.time()))
tf.contrib.saved_model.simple_save(sess, SAVEDMODEL,
                           inputs={"features":features, "Hin":Hin, "dropout_pkeep":dropout_pkeep},
                           outputs={"Yout":Yout, "H":H})

plt.ylim(ymax=np.amax(losses[1:])) # ignore first value for scaling
plt.plot(indices, losses)
plt.show()

def prediction_run(predict_fn, prime_data, run_length):
    H = np.zeros([1, RNN_CELLSIZE * N_LAYERS]) # zero state initially
    Yout = np.zeros([1, N_FORWARD, 2])
    data_len = prime_data.shape[0]-N_FORWARD

    # prime the state from data
    if data_len > 0:
        Yin = np.array(prime_data[:-N_FORWARD])
        Yin = np.reshape(Yin, [1, data_len, 2]) # reshape as one sequence of pairs (Tmin, Tmax)
        r = predict_fn({'features': Yin, 'Hin':H, 'dropout_pkeep':1.0}) # no dropout during inference
        Yout = r["Yout"]
        H = r["H"]
        
        # initaily, put real data on the inputs, not predictions
        Yout = np.expand_dims(prime_data[-N_FORWARD:], axis=0)
        # Yout shape [1, N_FORWARD, 2]: batch of a single sequence of length N_FORWARD of (Tmin, Tmax) data pointa
    
    # run prediction
    # To generate a sequence, run a trained cell in a loop passing as input and input state
    # respectively the output and output state from the previous iteration.
    results = []
    for i in range(run_length//N_FORWARD+1):
        r = predict_fn({'features': Yout, 'Hin':H, 'dropout_pkeep':1.0}) # no dropout during inference
        Yout = r["Yout"]
        H = r["H"]
        results.append(Yout[0]) # shape [N_FORWARD, 2]
        
    return np.concatenate(results, axis=0)[:run_length]

CHUNKSIZE = 150*100*2
#this RESAMPLE_BY is the size of your smoothing kernel? 
# -don't worry, seems feature extraction took care of this?
#worth a try later if training too slow/bad convergence on loss...
#sampling rate is 4GHZ???
# Try starting predictions from January / March / July (resp. OFFSET = YEAR or YEAR+QYEAR or YEAR+2*QYEAR)
# Some start dates are more challenging for the model than others.
#OFFSET = 30*YEAR+1*QYEAR
OFFSET = 30*CHUNKSIZE#coef of OFFSET upper bounded by 4196/2 - BATCH_LIMIT

PRIMELEN=5*YEAR#?something extra?

RUNLEN=BATCH_LIMIT*CHUNKSIZE
RMSELEN=3*CHUNKSIZE # accuracy of predictions next 3 chunks???

predict_fn = tf.contrib.predictor.from_saved_model(SAVEDMODEL)

for evaldata in valid_gen:
    prime_data = evaldata[OFFSET:OFFSET+PRIMELEN]
    results = prediction_run(predict_fn, prime_data, RUNLEN)
    picture_this(evaldata, evalranges, 
        prime_data, results, PRIMELEN, RUNLEN, OFFSET, RMSELEN)

def picture_these(evaldata, evalranges, prime_data, results, primelen, runlen, offset, rmselen):


def picture_this(evaldata, evalranges, prime_data, results, primelen, runlen, offset, rmselen):
    disp_data = evaldata[offset:offset+primelen+runlen]
    disp_dates = evaldates[offset:offset+primelen+runlen]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    displayresults = np.ma.array(np.concatenate((np.zeros([primelen,2]), results)))
    displayresults = np.ma.masked_where(displayresults == 0, displayresults)
    sp = plt.subplot(212)
    p = plt.fill_between(disp_dates, displayresults[:,0], displayresults[:,1])
    p.set_alpha(0.8)
    p.set_zorder(10)
    trans = plttrans.blended_transform_factory(sp.transData, sp.transAxes)
    plt.text(disp_dates[primelen],0.05,"DATA |", color=colors[1], horizontalalignment="right", transform=trans)
    plt.text(disp_dates[primelen],0.05,"| +PREDICTED", color=colors[0], horizontalalignment="left", transform=trans)
    plt.fill_between(disp_dates, disp_data[:,0], disp_data[:,1])
    plt.axvspan(disp_dates[primelen], disp_dates[primelen+rmselen], color='grey', alpha=0.1, ymin=0.05, ymax=0.95)
    plt.show()

    rmse = math.sqrt(np.mean((evaldata[offset+primelen:offset+primelen+rmselen] - results[:rmselen])**2))
    print("RMSE on {} predictions (shaded area): {}".format(rmselen, rmse))
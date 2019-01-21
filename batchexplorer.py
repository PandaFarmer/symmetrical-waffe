import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
import os


#df = df.pivot_table(index = ["time_to_failure", "acoustic_data"], aggfunc = 'count')
#group_t = df.groupby(by="time_to_failure").count()

#group_t.count
#df = df.apply(pd.value_counts).fillna(0)
#DIR = '/TrainSplits'
#num_batches = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
num_batches = 4194

def read_batch(batch_num):
    df = pd.read_csv("../KaggleLANL/TrainSplits/train_batch_num%d.csv"%batch_num)
    df = df.drop(columns=["Unnamed: 0"])
    return df

def plot_time_series():
    for batch_num in range(num_batches):
        df = read_batch(batch_num)

        fig = plt.figure(num=None, figsize=(13, 8), dpi=80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)

        sampling_freq = 4000000

        plt.plot(df.index/sampling_freq, df["acoustic_data"])
        plt.plot(df.index/sampling_freq, df["time_to_failure"]*100)
        plt.title('Acoustic Signal in Time Domain for Batch: %d'%batch_num)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.show()

def plot_freq_series():
    for batch_num in range(num_batches):
        df = read_batch(batch_num)

        np.random.seed(0)

        dt = 0.1  # sampling interval
        Fs = 1 / dt  # sampling frequency
        #t = np.arange(0, 10, dt)
        t = df.index/400000 #400kHZ to account for scaled sampling interval

        # generate noise:
        #nse = np.random.randn(len(t))
        nse = df["acoustic_data"]
        r = np.exp(-t / 0.05)
        cnse = np.convolve(nse, r) * dt
        cnse = cnse[:len(t)]

        s = 0.1 * np.sin(4 * np.pi * t) + cnse  # the signal

        '''
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(7, 7))

        # plot time signal:
        axes[0, 0].set_title("Signal")
        axes[0, 0].plot(t, s, color='C0')
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Amplitude")

        # plot different spectrum types:
        axes[1, 0].set_title("Magnitude Spectrum")
        axes[1, 0].magnitude_spectrum(s, Fs=Fs, color='C1')

        axes[1, 1].set_title("Log. Magnitude Spectrum")
        axes[1, 1].magnitude_spectrum(s, Fs=Fs, scale='dB', color='C1')

        axes[2, 0].set_title("Phase Spectrum ")
        axes[2, 0].phase_spectrum(s, Fs=Fs, color='C2')

        axes[2, 1].set_title("Angle Spectrum")
        axes[2, 1].angle_spectrum(s, Fs=Fs, color='C2')

        axes[0, 1].remove()  # don't display empty ax

        fig.tight_layout()
        '''
        plt.magnitude_spectrum(s, Fs=Fs, color='C1')
        plt.title('Acoustic Signal in Frequency Domain for Batch: %d'%batch_num)
        plt.xlabel('Frequency (hz)')
        plt.ylabel('Magnitude')

        plt.show()

plot_freq_series()


#print(df.ix[0])
#print(df.ix[4998])

#print(df.groupby(["time_to_failure"]).agg(['count']))
#print(pd.pivot_table(df, index=["time_to_failure"], aggfunc=['count']))
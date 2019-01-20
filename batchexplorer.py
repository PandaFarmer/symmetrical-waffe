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

def plot_time_series():
    for batch_num in range(4194):
        start = time.time()

        df = pd.read_csv("TrainSplits/train_batch_num%d.csv"%batch_num)
        end = time.time()
        print(end-start)
        df = df.drop(columns=["Unnamed: 0"])

        fig = plt.figure(num=None, figsize=(13, 8), dpi=80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)

        plt.plot(df.index/4000000, df["acoustic_data"])
        plt.plot(df.index/4000000, df["time_to_failure"]*100)
        plt.title('Acoustic Signal in Time Domain for Batch: %d'%batch_num)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.show()



#print(df.ix[0])
#print(df.ix[4998])

#print(df.groupby(["time_to_failure"]).agg(['count']))
#print(pd.pivot_table(df, index=["time_to_failure"], aggfunc=['count']))
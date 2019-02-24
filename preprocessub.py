import numpy as np
import pandas as pd
import time

chunksize = 150000

start_write = time.time()

for i, chunk in enumerate(pd.read_csv("input/train.csv", chunksize=chunksize)):
	start_batch_write = time.time()
	print("writing to file rows batchnum: %d"%(i), end="\r")
	chunk.ix[i*chunksize : i*chunksize + chunksize].to_csv(path_or_buf="input/train/train_batch_num%s.csv"%i)
	end_batch_write = time.time()
	print("Batch write completed for batch_num: %d in %.3f"%(i, end_batch_write-start_batch_write), end="\r")

end_write = time.time()

print("Write completed in: {}s".format(end_write - start_write))
import numpy as np
import pandas as pd
import time

start_read = time.time()
df = pd.read_csv("../KaggleLANL/train.csv")

#df_ss = pd.read_csv("KaggleLANL/sample_submission.csv")

end_read = time.time()

print("Read completed in: {}s".format(end_read - start_read))

start_write = time.time()
i = 0
batchsize = 150000
rows = df.shape[0]

while i*batchsize < rows:
	start_batch_write = time.time()
	print("writing to file rows batchnum: %d"%(i))
	df.ix[i*batchsize : i*batchsize + batchsize].to_csv(path_or_buf="../KaggleLANL/TrainSplits/train_batch_num%s.csv"%i)
	end_batch_write = time.time()
	print("Batch write completed for batch_num: %d in %ds"%(i, end_batchwrite-start_batch_write))
	i += 1
'''
print("writing to file rows %d to %d"%(i*batchsize, rows - 1))
df.ix[i*batchsize : rows].to_csv(path_or_buf="TrainSplits/train_batch_num%s.csv"%i)
'''
end_write = time.time()

#print(df.shape)
#print(df.head())
print("Write completed in: {}s".format(end_write - start_write))


#print(df_ss)
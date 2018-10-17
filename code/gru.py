#!/usr/bin/env python

import os, sys
import numpy as np
import matplotlib as mpl
mpl.use('AGG')

import matplotlib.pyplot as plt
from keras import layers
from keras import initializers
from keras.models import Sequential
from keras.optimizers import RMSprop


### 6.28 Inspecting the Price/Volatility data
# see datasets dict below
dset = str(sys.argv[1])
epoch_steps = int(sys.argv[2])
eps = int(sys.argv[3])

# data_dir = '/Users/sanch/Dropbox/f2018/CIS693/cis693_ms_project/data'
data_dir = '/home/sanchrob/repos/cis693_ms_project/data'
datasets = {'rt_bomo': 'PJM_rt_bomo.csv',
            'da_bomo': 'PJM_da_bomo.csv',
            'rt_before': 'PJM_rt_before.csv',
            'da_before': 'PJM_da_before.csv',
            'rt_after': 'PJM_rt_after.csv',
            'da_after': 'PJM_da_after.csv',
            'rt_combined': 'PJM_rt_combined.csv',
            'da_combined': 'PJM_da_combined.csv'}

fname = os.path.join(data_dir, datasets.get(dset, 'default'))

f = open(fname)
data = f.read()
f.close()

# splitlines() discards last empty line (universal newlines)
raw_lines = data.splitlines()
header = raw_lines[0].split(',')
lines = raw_lines[1:]

# print(header)
# print(len(lines))


### 6.29 Parsing the data
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values
#   print(i, line, values)


### 6.30 Plotting the price & volatility timeseries
#### scale plots larger
def scale_plot_size(factor=1.10):
    default_dpi = mpl.rcParamsDefault['figure.dpi']
    mpl.rcParams['figure.dpi'] = default_dpi*factor
    
scale_plot_size()



price = float_data[:, 0]
volatility = float_data[:, 1]

# blue
plt.plot(range(len(price)), price)
# orange
plt.plot(range(len(volatility)), volatility)
# plt.savefig('timeseries.png')


### 6.31 Plotting a narrower window of Price/Volatility
# hourly timesteps, 48 = two days
plt.plot(range(200), price[300:500])
plt.plot(range(200), volatility[300:500])
# plt.savefig('window.png')

#raw_lines[325]


### 6.32 Normalization
# normalize only the training data?
# 120 obs: 50% train, 25% val, 25% test
mean = float_data[:60].mean(axis=0)
float_data -= mean
std = float_data[:60].std(axis=0)
float_data /= std

# get shape of the matrix
float_data.shape


### 6.33 Generator for yielding timeseries samples and targets

# lookback = 24; observations will go back 1 day
# steps = 1; observations will be sampled hourly
# delay = 12; targets wil be 12 hours in the future

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=6, step=1):
    
    # only applies to test_gen
    if max_index is None:
        max_index = len(data) - delay - 1
    
    i = min_index + lookback
#     print("init, i: ", i)
      
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
                        
            rows = np.arange(i, min(i + batch_size, max_index))
#             print(rows, len(rows))
#             print("i: ", i)
            i += len(rows)
#             print("increment i: ", i)
                
        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]  # field [1] is volatility, the response 
#            targets[j] = data[rows[j] + delay][0]  # field [0] is price, a predictor
        
        yield samples, targets


### 6.34 Prepare train, validation, test generators

batch_size = 24 # number of samples (of size lookback) per batch
lookback = 24   # sample size (history)
delay = 24      # target (predict n + delay steps from now)
step = 1        # sample frequency (1 = hourly)


# 70%
tr_max_idx = int((len(float_data) * 0.7) // 1) - 1
#print("  tr_max_idx: ", tr_max_idx)
train_gen = generator(float_data,
                      lookback=lookback, 
                      delay=delay, 
                      min_index=0, 
                      max_index=tr_max_idx,
                      shuffle=False,
                      step=step, 
                      batch_size=batch_size)

# 15%
val_min_idx = tr_max_idx + 1
val_max_idx = val_min_idx + int((len(float_data) * 0.15) // 1) - 1
#print(" val_min_idx: ", val_min_idx)
#print(" val_max_idx: ", val_max_idx)
val_gen = generator(float_data, 
                    lookback=lookback, 
                    delay=delay, 
                    min_index=val_min_idx, 
                    max_index=val_max_idx, 
                    step=step, 
                    batch_size=batch_size)

# 15%
test_min_idx = val_max_idx + 1
#print("test_min_idx: ", test_min_idx)
test_gen = generator(float_data, 
                     lookback=lookback, 
                     delay=delay, 
                     min_index=test_min_idx, 
                     max_index=None, 
                     step=step, 
                     batch_size=batch_size)

val_steps = (val_max_idx - val_min_idx - lookback)
test_steps = (len(float_data) - test_min_idx - lookback)


### 6.35 Computing common-sense baseline MAE (mean absolute error)

def eval_naive_mae():
    batch_maes = []
    for step in range(val_steps):
#    for step in range(test_steps):
        samples, targets = next(val_gen)
#        samples, targets = next(test_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    
    return np.mean(batch_maes)

# store Naive perf. measure
naive_mae = eval_naive_mae()

def eval_naive_mse():
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mse = np.sum((preds - targets)**2)
        batch_mses.append(mse)

### 6.36 Convert MAE back to Volatility error


# multiply result of naive method above (~1.5) x SD(volatility)
volatility_mae = naive_mae * std[1]
#print(volatility_mae)


### 6.3.9 Train and evaluate a GRU base-model



model = Sequential()
model.add(layers.GRU(64,
                     kernel_initializer=initializers.glorot_normal(seed=1337),
                     recurrent_initializer=initializers.orthogonal(seed=1337),
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
# n - k + 1
history = model.fit_generator(train_gen,
                              steps_per_epoch = epoch_steps,
                              epochs = eps,
                              validation_data=val_gen,
                              validation_steps=val_steps)

### 6.38 Plot results


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss (MAE)')
plt.legend()

#plt.show()
figname = str(epoch_steps) + '_' + str(eps) + '_' + dset + '.png'
plt.savefig(figname)

print("\nmean training loss: ", np.mean(loss).round(4))
print("mean validation loss: ", np.mean(val_loss).round(4))
print("Naive MAE: ", round(naive_mae, 4))


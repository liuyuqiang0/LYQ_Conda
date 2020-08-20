import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics
import os,wget,csv



def Get_Data(url):
    if 'Sunspots.csv' not in os.listdir('.'):wget.download(url)
    return pd.read_csv('Sunspots.csv')

def plot_series(ax,x,y,title=None):
    ax.plot(x, y)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(title)


data_set=Get_Data('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/Sunspots.csv')
# print(data_set.info())
# print(data_set.head())

sunspots=np.array(data_set['Monthly Mean Total Sunspot Number'])
time_step=np.arange(len(sunspots))

# fig,ax=plt.subplots(figsize=(10,6))
# plot_series(ax,time_step,sunspots)
# plt.show()

split_time = 3000
time_train = time_step[:split_time]
x_train = sunspots[:split_time]
time_valid = time_step[split_time:]
x_valid = sunspots[split_time:]

window_size=30
batch_size=100
shuffle_buffer_size=1000
epochs=300
learn_rate=1e-5

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series) # Creates a Dataset whose elements are slices of the given tensors.
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))  # add a label
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1) # 最后一批数量可以不足
    return dataset

def model_forecast(model, series, window_size,batch_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size).prefetch(1)
    forecast = model.predict(ds)
    return forecast

dataset=windowed_dataset(x_train,window_size,batch_size,shuffle_buffer_size)

# Combined Network
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=64,
                           kernel_size=5,
                           strides=1,
                           padding='causal',
                           input_shape=[None,1],
                           activation='relu'),
    tf.keras.layers.LSTM(64,return_sequences=True),
    tf.keras.layers.LSTM(64,return_sequences=True),
    tf.keras.layers.Dense(30,input_shape=[window_size],activation='relu'),
    tf.keras.layers.Dense(10,activation='relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x:x*400)
])

# lr_schedule=tf.keras.callbacks.LearningRateScheduler(lambda epoch:1e-8 * 10**(epoch/20))

optimizer=tf.keras.optimizers.SGD(lr=learn_rate,momentum=0.9)
model.compile(loss='mae',optimizer=optimizer)
history=model.fit(dataset,epochs=epochs,verbose=2)

# forecast=[]
# for time in range(split_time-window_size,len(sunspots) - window_size):
#     forecast.append(model.predict(sunspots[time:time + window_size][np.newaxis]))

results=model_forecast(model, sunspots[..., np.newaxis], window_size,batch_size)
results = results[split_time - window_size:-1, -1, 0]

fig,ax=plt.subplots(figsize=(10, 6))

mae=tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
plot_series(ax,time_valid, x_valid)
plot_series(ax,time_valid, results,('Combined Network: mae = %.6f, window_size = %d, learn_rate = %g, epochs= %d' %(mae,window_size,learn_rate,epochs)))
plt.show()
print(mae)

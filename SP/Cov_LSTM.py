import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics

def plot_series(ax,time, series, title=None,format="-", start=0, end=None):
    ax.plot(time[start:end], series[start:end], format)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True)

def plot_mae_loss(ax,epochs,mae,loss):
    ax.plot(epochs, mae,'r')
    ax.plot(epochs,loss,'b')
    ax.set_title('MAE and Loss')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend(["MAE", "Loss"])

def trend(time, slope=0.0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4 * 365 + 1, dtype="float32")  # 生成4个周期，一个周期一年
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size=30
batch_size=48
shuffle_buffer_size=1000


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series=tf.expand_dims(series,-1)
    dataset = tf.data.Dataset.from_tensor_slices(series) # Creates a Dataset whose elements are slices of the given tensors.
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))  # add a label
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))

    return dataset.batch(batch_size).prefetch(1)

def model_forecast(model, series, window_size,batch_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size).prefetch(1)
    forecast = model.predict(ds)
    return forecast

train_data=windowed_dataset(x_train,window_size,batch_size=128,shuffle_buffer=shuffle_buffer_size)

model=tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32,kernel_size=5,
                           strides=1,padding='causal',
                           activation='relu',
                           input_shape=[None,1]),
    # tf.keras.layers.LSTM(32, return_sequences=True),
    # tf.keras.layers.LSTM(32, return_sequences=True),
    keras.layers.Bidirectional(tf.keras.layers.LSTM(32,return_sequences=True)),
    keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x:x*200)
])
model.summary()

optimizer=tf.keras.optimizers.SGD(lr=3e-5,momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=['mae'])

# lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch: 1e-8 * 10**(epoch / 20))

history=model.fit(train_data,epochs=400,verbose=0)

# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-8, 1e-4, 0, 30])
# plt.show()

results=model_forecast(model, series[..., np.newaxis], window_size,batch_size)
print(results.shape)
results=results.flatten()[split_time - window_size:-1]



fig,ax=plt.subplots(figsize=(10, 6))

plot_series(ax,time_valid, x_valid)
plot_series(ax,time_valid, results)
plt.show()
print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())

mae=history.history['mae']
loss=history.history['loss']
epochs=range(len(loss))

fig,ax=plt.subplots(figsize=(18,8),nrows=1,ncols=2)
plot_mae_loss(ax[0],epochs,mae,loss)
plot_mae_loss(ax[1],epochs[200:],mae[200:],loss[200:])  # 放大
plt.show()


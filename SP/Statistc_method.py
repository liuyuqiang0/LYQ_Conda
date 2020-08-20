import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import metrics


def plot_series(ax,time, series, title=None,format="-", start=0, end=None):
    ax.plot(time[start:end], series[start:end], format)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True)

def trend(time, slope=0):
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
# series = trend(time, 0.1)
amplitude = 40
slope = 0.05
noise_level = 5

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# fig,ax=plt.subplots(figsize=(20, 6),nrows=1,ncols=2)
# plot_series(ax[0],time, series,'Without noise')

# Update with noise
series += noise(time, noise_level, seed=42)

# plot_series(ax[1],time, series,'With noise')
# fig.suptitle('Time Series',size=20)
# plt.show()


split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

# fig,ax=plt.subplots(figsize=(18, 8),nrows=1,ncols=2)
# plot_series(ax[0],time_train, x_train,'Trainning_Series')
# plot_series(ax[1],time_valid, x_valid,'Validation_Series')
# fig.suptitle('Split Datasets',size=20,weight='bold')
# plt.show()


naive_forecast = series[split_time - 1:-1]
# fig,ax=plt.subplots(figsize=(10, 6))
# plot_series(ax,time_valid, x_valid)
# plot_series(ax,time_valid, naive_forecast)
# fig.suptitle('Naive predict',size=20,weight='bold')
# plt.show()


# fig,ax=plt.subplots(figsize=(10, 6))
# plot_series(ax,time_valid, x_valid, start=0, end=150)
# plot_series(ax,time_valid, naive_forecast, start=1, end=151)
# plt.suptitle('Naive predict',size=20,weight='bold')
# plt.show()

# print(metrics.MSE(x_valid,naive_forecast))
# print(metrics.MAE(x_valid,naive_forecast))


def moving_average_forecast(series, window_size):
  """Forecasts the mean of the last few values.
     If window_size=1, then this is equivalent to naive forecast"""
  forecast = []
  for time in range(len(series) - window_size):
    forecast.append(series[time:time + window_size].mean())
  return np.array(forecast)

moving_avg = moving_average_forecast(series, 30)[split_time - 30:]

# fig,ax=plt.subplots(figsize=(10, 6))
# plot_series(ax,time_valid, x_valid)
# plot_series(ax,time_valid, moving_avg)
# plt.suptitle('Moving_avg',size=20,weight='bold')
# plt.show()
# print(metrics.MSE(x_valid,moving_avg).numpy())
# print(metrics.MAE(x_valid,moving_avg).numpy())


diff_series = (series[365:] - series[:-365]) # 周期为一年，所以需要从一年开始计算
diff_time = time[365:]

# fig,ax=plt.subplots(figsize=(10, 6))
# plot_series(ax,diff_time, diff_series)
# plt.suptitle('Diff_series',size=18,weight='normal')
# plt.show()


diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50:]  # 得出差分序列(总长度减去了365)的移动平均线后，为了保证预测长度和验证集长度一样，不仅要减去之前的365还要减去50的窗口大小

# fig,ax=plt.subplots(figsize=(10, 6))
# plot_series(ax,time_valid, diff_series[split_time - 365:])
# plot_series(ax,time_valid, diff_moving_avg)
# plt.suptitle('Diff_moving_avg',size=18,weight='normal')
# plt.show()


diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg
#
# fig,ax=plt.subplots(figsize=(10, 6))
# plot_series(ax,time_valid, x_valid)
# plot_series(ax,time_valid, diff_moving_avg_plus_past)
# plt.suptitle('Diff_moving_avg_plus_past',size=18,weight='normal')
# plt.show()
#
# print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_past).numpy())
# print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy())



diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-360], 10) + diff_moving_avg

fig,ax=plt.subplots(figsize=(10, 6))
plot_series(ax,time_valid, x_valid)
plot_series(ax,time_valid, diff_moving_avg_plus_smooth_past)
plt.suptitle('diff_moving_avg_plus_smooth_past',size=18,weight='normal')
plt.show()

print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import metrics

def plot_series(time, series):
    plt.figure(figsize=(10, 6))
    plt.plot(time, series)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.grid(True)
    plt.show()


def autocorrelation(time, amplitude):
    rho1 = 0.5
    rho2 = -0.1
    ar = np.random.randn(len(time) + 50)
    ar[:50] = 100
    for step in range(50, len(time) + 50):
        ar[step] += rho1 * ar[step - 50]
        ar[step] += rho2 * ar[step - 33]
    return ar[50:] * amplitude


time = np.arange(4 * 365 + 1)
series = autocorrelation(time, 10)
plot_series(time[:200], series[:200])



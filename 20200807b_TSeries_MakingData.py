



import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


####################################################
# 데이터 생성함수 정의
####################################################

def plot_series(time, series, format="-", start=0, end=None, legend=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    if legend:
        plt.legend(fontsize=14)
    plt.grid(True)


# 기울기
def trend(time, slope=0):
    return slope * time


# 1년마다 나타나는 현상
# numpy.where()
# season_time<0.4 이면 np.cos(season_tim*2*np.pi) 아니면 1/np.exp(3*season_time)
def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where( season_time<0.4,
                     np.cos(season_time*2*np.pi),
                     1/(np.exp(3*season_time)) )

def seasonality( time, period, amplitude=1, phase=0 ):
    """Repeats the same pattern at each period"""
    season_time = ( (time+phase)%period)/period
    return amplitude*seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


# autocorrelation #1
def autocorrelation( time, amplitude ):
    rho1 = 0.5
    rho2 = -0.1
    ar = np.random.randn( len(time)+50 )
    ar[:50] = 100
    for step in range(50, len(time) + 50):
        ar[step] += rho1 * ar[step-50]
        ar[step] += rho2 * ar[step-33]
    return ar[50:] * amplitude


# autocorrelation #2
def autocorrelation( time, amplitude ):
    rho = 0.8
    ar = np.random.randn( len(time)+1 )
    for step in range(50, len(time) + 1):
        ar[step] += rho * ar[step-1]
    return ar[1:] * amplitude

def autocorelation( source, Φs ):
    ar = source.copy()
    max_lag = len( Φs )
    for step, value in enumerate( source ):
        for lag, Φ in Φs.items():
            if step - lag > 0:
                ar[step] += Φ * ar[step-lag]
    return ar


def impulses( time, num_impulses, amplitude=1, seed=None):
    rnd = np.random.RandomState( seed )
    impulse_indices = rnd.randint( len(time), size=10 )
    series = np.zeros( len(time) )
    for index in impulse_indices:
        series[index] += rnd.rand() + amplitude
    return series


####################################################
# 데이터 생성 파라미터
####################################################
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5
time = np.arange(4 * 365 + 1, dtype="float32")


####################################################
# 그래프
####################################################

# Trend
series = trend( time, 0.1 )
plot_series(time, series)

# baseline + Trend
series = baseline + trend( time, 0.1 )
plot_series(time, series)


# noise
series = baseline + trend(time, slope) + noise(time, noise_level, seed=42)
plot_series(time, series)


# Seasonality
series = seasonality(time, period=365, amplitude=amplitude)
plot_series(time, series)


# autocorrrelation
series = autocorrelation(time, 10)
plot_series(time, series)
plt.figure()
plot_series(time[:200], series[:200])

# noise
series = noise( time )
plot_series(time, series)
plt.figure()
plot_series(time[:200], series[:200])


# 기울기 + Seasonality + noise
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
series += noise(time, noise_level, seed=42)
plot_series(time, series)
plt.show()

# 두개 결합
series =        trend(time, 2) + autocorrelation(time, 10) + seasonality(time, period=365, amplitude=150)
series2 = 550 + trend(time, -1) + autocorrelation(time, 5) + seasonality(time, period=365, amplitude=2)
series[200:] = series2[200:]
series = series + noise(time, 30)
plot_series(time, series)
plt.figure()
plot_series(time[:300], series[:300])


# impulse
series = impulses(time, 10, seed=42)
plot_series(time, series)

# impuse + correlation
signal = impulses(time, 10, seed=42)
series = autocorelation(signal, {1:0.99})
plot_series(time, series)
plt.plot(time, signal, 'k-')

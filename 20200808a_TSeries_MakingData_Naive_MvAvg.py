



import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


####################################################
# 데이터 생성함수 정의
####################################################

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
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

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5


####################################################
# 그래프
####################################################

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)

# 그래프 : 기울기 + Seasonality + noise
plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()


split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

plt.figure(figsize=(10, 6))
plot_series(time_train, x_train)
plt.show()

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plt.show()

###################################
# Naive Forecast
###################################

naive_forecast = series[split_time-1:-1] #시간지연 1 시킴

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, naive_forecast)

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, start=0, end=150)
plot_series(time_valid, naive_forecast, start=1, end=151)

# keras.metrics.mean_squared_error
# keras.metrics.mean_absolute_error
print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())


###################################
# moving average
###################################


def moving_average_forecast(series, window_size):
  """Forecasts the mean of the last few values.
     If window_size=1, then this is equivalent to naive forecast"""
  forecast = [] # list
  for time in range(len(series) - window_size):
    forecast.append(series[time:time+window_size].mean())
  return np.array(forecast) # list -> nparry 변환

moving_avg = moving_average_forecast(series, 30)[split_time-30:] # moving_average_forecast(series, 30) -> nparray 그래서 nparry[split_time-30:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)



print(keras.metrics.mean_squared_error(x_valid, moving_avg).numpy())
# keras.metrics.mean_squared_error(x_valid, moving_avg) 를 바로 표시할 수 없음.
# type( keras.metrics.mean_squared_error(x_valid, moving_avg) :  tensorflow.python.framework.ops.EagerTensor
# mse = keras.metrics.mean_squared_error(x_valid, moving_avg )
# mse.numpy()

print(keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy())


#
# trend 와 seasonality 제거하기
#
diff_series = (series[365:] - series[:-365])
diff_time = time[365:]

plt.figure(figsize=(10, 6))
plot_series(diff_time, diff_series)
plt.show()


# trend 와 seasonality 제거 후 Moving Average
diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50:] # 365는 seasonlity 주기, 50은 MV의 Window

plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - 365:])
plot_series(time_valid, diff_moving_avg)
plt.show()


diff_moving_avg_plus_past = series[split_time-365:-365] + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_past)
plt.show()



print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_past).numpy())
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy())


diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-360], 10) + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_smooth_past)
plt.show()

print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
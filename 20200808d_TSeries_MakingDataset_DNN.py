



import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


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

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  # from_tensor_slices 클래스 메서드를 사용하면 리스트, 넘파이, 텐서플로 자료형에서 데이터셋을 만들 수 있다.
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  # window의 맨 마지막은 label, 나머지는 train_data
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset


####################################################
# 데이터 생성 파라미터
####################################################

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000


# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)



#############################################
# Data Processing
#############################################
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]


plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.title( 'Train + Validation' )
plt.show()

plt.figure(figsize=(10, 6))
plot_series(time_train, x_train)
plt.title( 'Train' )
plt.show()

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plt.title( 'Validation' )
plt.show()

##############################################################
# Model -> Training -> Learning Curve : Tuning LearningRate
##############################################################
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
print(dataset)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9), metrics=['accuracy'])
# 일반적으로 momentum은 0.9 사용
model.summary()

history = model.fit(dataset,epochs=100,verbose=1)

loss = history.history['loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.show()



##############################################################
# Model -> Training -> Learning Curve : Tuning LearningRate
##############################################################
window_size = 20
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)
])


optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss="mse", optimizer=optimizer)

lr_schedule = tf.keras.callbacks.LearningRateScheduler( lambda epoch: 1e-8 * 10**(epoch / 20) )
history = model.fit(dataset, epochs=100, callbacks=[lr_schedule], verbose=1)

lrs = 1e-8 * (10 ** (np.arange(100) / 20))
plt.semilogx(lrs, history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 300])



###############################
# Model
###############################
window_size = 30
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation="relu", input_shape=[window_size]),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.SGD(lr=8e-6, momentum=0.9) # <- 여기를 수정
model.compile(loss="mse", optimizer=optimizer)
###############################
# Training
###############################
history = model.fit(dataset, epochs=500, verbose=1)

###############################
# Learning Curve
###############################
loss = history.history['loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.show()


# Plot all but the first 10
loss = history.history['loss']
epochs = range(10, len(loss))
plot_loss = loss[10:]
print(plot_loss)
plt.plot(epochs, plot_loss, 'b', label='Training Loss')
plt.show()



###############################
# Prediction using the Model
###############################

forecast = []

for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:,0,0]

plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)
plot_series(time_valid, results)


tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
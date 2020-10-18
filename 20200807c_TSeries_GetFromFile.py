
import numpy as np
import matplotlib.pyplot as plt

def plot_series(time, series, format="-", start=0, end=None, legend=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    if legend:
        plt.legend(fontsize=14)
    plt.grid(True)

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



# impuse + correlation
time = np.arange(4 * 365 + 1, dtype="float32")
signal = impulses(time, 10, seed=42)
series = autocorelation(signal, {1:0.99})
plot_series(time, series)
plt.plot(time, signal, 'k-')



# ARIMA : Autoregressive integrated moving average
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(series, order=(5,1,0))
model_fit = model.fit(disp=0)
print( model_fit.summary() )



import pandas as pd


df = pd.read_csv('sunspots.csv', parse_dates=['Date'], index_col='Date') # 'Data'열은 날짜로 Parsing하고, 'Data'열이 index_col에 해당하는 열
series = df['Monthly Mean Total Sunspot Number'].asfreq('1M')
series.head()

type( series )

series.plot( )

series['1995-01-01':].plot()


'''
s = pd.Series([1, 1, 2, 3, 5, 8])
s.diff() # = s.diff(1)

0    NaN
1    0.0
2    1.0
3    1.0
4    2.0
5    3.0
dtype: float64

s.diff(periods=-1)
0    0.0
1   -1.0
2   -1.0
3   -2.0
4   -3.0
5    NaN
dtype: float64
'''
series.diff(1) # index는 두고, column(0) 간의 차이

series.diff(1).plot() # index는 두고, column(0) 간의 차이를 그래프로


# Autocorrelation plot
# Correlation 은 두 변수 간의 선형적 관계를 측정하고자 할 때 사용한다. 그런데 autocorrelation 은 자기 자신 (auto) 과 correlation 을 알아보고자 할 때 사용한다.
# 자기 자신과의 관계이기 때문에 lag 된 값을 이용한다.
# 시계열로 한정하는 경우 x축에 시차(lag), y축에 상관계수(correlation coefficient)를 놓은 뒤 plotting한 것을 의미하게 된다.
# Autocorrelation plots are often used for checking randomness in time series.
# This is done by computing autocorrelations for data values at varying time lags.
# If time series is random, such autocorrelations should be near zero for any and all time-lag separations.
# If time series is non-random then one or more of the autocorrelations will be significantly non-zero.
# The horizontal lines displayed in the plot correspond to 95% and 99% confidence bands.
# The dashed line is 99% confidence band. See the Wikipedia entry for more about autocorrelation plots.
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(series)

autocorrelation_plot(( series.diff(1)[1:]))
plt.axis( [0, 500, -0.1, 0.1])

autocorrelation_plot(series.diff(1)[1:].diff(11 * 12)[11*12+1:])
plt.axis([0, 500, -0.1, 0.1])

autocorrelation_plot(series.diff(1)[1:])
plt.axis([0, 50, -0.1, 0.1])


from pandas.plotting import autocorrelation_plot
series_diff = series
for lag in range(50):
  series_diff = series_diff[1:] - series_diff[:-1]
autocorrelation_plot(series_diff)

#  아래는 그림이 안 그려졌음
import pandas as pd
series_diff1 = pd.Series(series[1:] - series[:-1])
autocorrs = [series_diff1.autocorr(lag) for lag in range(1, 60)]
plt.plot(autocorrs)
plt.show()
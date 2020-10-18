

# TSeres_MakingDataset_1Layer 에서 사용


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)


dataset = tf.data.Dataset.range(10)
for val in dataset:
   print(val.numpy())


dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1) # 5개씩 window 를 만드는데, shift 1씩 해서 윈도우 구성이 1개까지 반복
for window_dataset in dataset:
  for val in window_dataset:
    print(val.numpy(), end=" ") # 한글자 마다 끝은 " "로, 이는 "0 1 2 3 4" 일케 만듬
  print() # 다음 줄
'''
0 1 2 3 4 
1 2 3 4 5 
2 3 4 5 6 
3 4 5 6 7 
4 5 6 7 8 
5 6 7 8 9 
6 7 8 9 
7 8 9 
8 9 
9 
'''

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True) # <- 위와 차이점
for window_dataset in dataset:
  for val in window_dataset:
    print(val.numpy(), end=" ")
  print()
'''
0 1 2 3 4 
1 2 3 4 5 
2 3 4 5 6 
3 4 5 6 7 
4 5 6 7 8 
5 6 7 8 9 
'''

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5)) # <- 위와 차이점
for window in dataset:
  print(window.numpy())
'''
[0 1 2 3 4]
[1 2 3 4 5]
[2 3 4 5 6]
[3 4 5 6 7]
[4 5 6 7 8]
[5 6 7 8 9]
'''


dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:])) # <- 위와 차이점
for x,y in dataset:
  print(x.numpy(), y.numpy())
'''
[0 1 2 3] [4]
[1 2 3 4] [5]
[2 3 4 5] [6]
[3 4 5 6] [7]
[4 5 6 7] [8]
[5 6 7 8] [9]
'''

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10) # <- 위와 차이점
for x,y in dataset:
  print(x.numpy(), y.numpy())
'''
[0 1 2 3] [4]
[1 2 3 4] [5]
[5 6 7 8] [9]
[4 5 6 7] [8]
[3 4 5 6] [7]
[2 3 4 5] [6]
'''


dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)
for x,y in dataset:
  print("x = ", x.numpy())
  print("y = ", y.numpy())
'''
x =  [[4 5 6 7]
      [3 4 5 6]]
y =  [[8]
      [7]]
x =  [[1 2 3 4]
      [2 3 4 5]]
y =  [[5]
      [6]]
x =  [[5 6 7 8]
      [0 1 2 3]]
y =  [[9]
      [4]]
'''
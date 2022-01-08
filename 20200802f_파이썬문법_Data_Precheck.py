# import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# mnist 데이터 가져옴
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# 데이터 확인
print( "training_images[0]:", training_images[0] ) # image[0] 값
print( "training_labels[0]:", training_labels[0] ) # label[0] 값
plt.imshow( training_images[0] ) # ipynb에서는 바로 표시됨
plt.show() # *.py 에서는 show() 해야 표시됨

print( "type(training_images):", type(training_images)  ) # [list] or (tuple) or {dictionary} or set 인지 확인
print( "training_images.shape:", training_images.shape )  # 전체 사이즈 확인

num    = training_images.shape[0]
size_x = training_images.shape[1]
size_y = training_images.shape[2]
print( "num size_x size_y:", num, size_x, size_y )

print( "len of training_images:", len( training_images ) ) #갯수
print( "training_images[0].shape:", training_images[0].shape )  #이미자 한개의 사이즈


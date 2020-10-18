"""
Image Hello world
"""
#import module
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

np.set_printoptions( linewidth=200 )

plt.imshow( training_images[0] ) # <- *.py에서는 안됨. *.ipynb에서는 동작
print( training_images[0] )
print( training_labels[0] )


training_images = training_images / 255.0
test_images     = test_images / 255.0



layer_0 = tf.keras.layers.Flatten()
layer_1 = tf.keras.layers.Dense( 128, activation=tf.nn.relu ) #128:node num
layer_2 = tf.keras.layers.Dense( 10, activation=tf.nn.softmax ) #10:softmax로 구분되기 전의 node수 10개
model = tf.keras.models.Sequential( [layer_0, layer_1, layer_2] )

model.compile( optimizer=tf.optimizers.Adam(),
               loss="sparse_categorical_crossentropy",
               metrics=['accuracy'] )

model.fit( training_images, training_labels, epochs=5 )


print(  model.evaluate( test_images, test_labels ) ) #test_loss, test_acc#

classifications = model.predict( test_images )
print( classifications[0] )
print( test_labels[0] )
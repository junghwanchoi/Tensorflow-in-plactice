
#import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



#종료 callback을 만듬
class myCallback( tf.keras.callbacks.Callback ):
    def on_epoch_end(self, epoch, logs=None):
        if( logs.get('loss')<0.1 ):
            print("\nReached 90% accuracy so cancelling training!")
            self.stopped_epoch = epoch
            self.model.stop_training = True

###############################
# 데이터 가져오기
###############################

#mnist 데이터 가져옴
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

###############################
# 데이터 확인
###############################

#mnist 데이터 그리고, 사이즈 확인
plt.imshow( training_images[0] )
print( training_labels[0] )
print( training_images[0] )

###############################
# Data Processing
###############################

#Normalize
training_images = training_images/255
test_images     = test_images/255

###############################
# Model
###############################

#layer를 만들고, 연결해서 NN 만듬
layer_0 = tf.keras.layers.Flatten()
layer_1 = tf.keras.layers.Dense( 128, activation=tf.nn.relu )
layer_2 = tf.keras.layers.Dense( 10, activation=tf.nn.softmax )
model = tf.keras.Sequential( [layer_0, layer_1, layer_2] )
model.compile( optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'] )

###############################
# Training
###############################

#Training
history = model.fit( training_images, training_labels, epochs=5 )


###############################
# Learning Curve
###############################

import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
#    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
#    plt.legend([string, 'val_' + string])
    plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

###############################
# Prediction using the Model
###############################

#TEST 및 결과확인
test_loss, test_acc = model.evaluate( test_images, test_labels )
classification = model.predict( test_images )
print( classification[0] )
print( test_labels[0] )

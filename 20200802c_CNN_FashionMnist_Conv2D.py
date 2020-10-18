"""
Image Hello world
"""
#import module
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#종료 callback을 만듬 : 이 예제에서 callback 동작안함
class myCallback( tf.keras.callbacks.Callback ):
    def on_epoch_end(self, epoch, logs=None):
        if( logs.get('loss')<0.1 ): #0.1-90% accuracy
            print("\nReached x% accuracy so cancelling training!")
            self.stopped_epoch = epoch
            self.model.stop_training = True

fashion_mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

plt.imshow( training_images[0] )
print( training_images[0] )
print( training_labels[0] )
print( training_images[0].shape ) #1개 이미지 사이즈
print( len(training_images) )     #갯수

training_images = training_images.reshape( 60000, 28, 28, 1 )
training_images = training_images / 255.0
test_images     = test_images.reshape( 10000, 28, 28, 1 )
test_images     = test_images / 255.0

model = tf.keras.models.Sequential( [
    tf.keras.layers.Conv2D( 64, (3,3), activation='relu', input_shape=(28, 28, 1) ),
    tf.keras.layers.MaxPooling2D( 2,2 ),
    tf.keras.layers.Conv2D( 64, (3,3), activation='relu' ),
    tf.keras.layers.MaxPooling2D( 2,2 ),
    tf.keras.layers.Flatten( ),
    tf.keras.layers.Dense( 128, activation='relu' ), #128:node num
    tf.keras.layers.Dense( 10, activation='softmax' ) #10:softmax로 구분되기 전의 node수 10개
])

model.compile( optimizer=tf.optimizers.Adam(),
               loss="sparse_categorical_crossentropy",
               metrics=['accuracy'] )

model.summary( )

model.fit( training_images, training_labels, epochs=10, callbacks=[myCallback()] )

test_loss, test_acc = model.evaluate( test_images, test_labels )

classifications = model.predict( test_images )
print( classifications[0] )
print( test_labels[:10] )


#import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#종료 callback을 만듬
class myCallback( tf.keras.callbacks.Callback ):
    def on_epoch_end(self, epoch, logs=None):
        if( logs.get('loss')<0.01 ): #0.1:90%, 0.01:99%
            print("\nReached x% accuracy so cancelling training!")
            self.stopped_epoch = epoch
            self.model.stop_training = True


#mnist 데이터 가져옴
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#mnist 데이터 그리고, 사이즈 확인
plt.imshow( training_images[0] )
print( "training_labels[0]:", training_labels[0] )
#print( training_images[0] )

print( "len of training_images:", len( training_images ) ) #갯수
print( "")
print( "training_images[0].shape:", training_images[0].shape )  #이미자 한개의 사이즈


#Normalize
training_images = training_images/255
test_images     = test_images/255

training_images = training_images.reshape( 60000, 28, 28, 1 )
training_images = training_images / 255.0
test_images     = test_images.reshape( 10000, 28, 28, 1 )
test_images     = test_images / 255.0

model = tf.keras.models.Sequential( [
    tf.keras.layers.Conv2D( 64, (3,3), activation='relu', input_shape=(28, 28, 1) ),
    tf.keras.layers.MaxPooling2D( 2,2 ),
    tf.keras.layers.Flatten( ),
    tf.keras.layers.Dense( 128, activation='relu' ), #128:node num
    tf.keras.layers.Dense( 10, activation='softmax' ) #10:softmax로 구분되기 전의 node수 10개
])

model.compile( optimizer='adam',
               loss="sparse_categorical_crossentropy",
               metrics=['accuracy'] )

model.summary( )

model.fit( training_images, training_labels, epochs=10, callbacks=[myCallback()] )

test_loss, test_acc = model.evaluate( test_images, test_labels )

classifications = model.predict( test_images )
print( classifications[0] )
print( test_labels[:10] )

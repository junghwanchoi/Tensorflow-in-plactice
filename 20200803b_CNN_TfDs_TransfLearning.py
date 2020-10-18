import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds


tfds.disable_progress_bar()

###############################
# 데이터 다운로드
###############################

# TFDS 사용시, Parameter 및 Return값
(raw_train, raw_validation, raw_test), metadata = \
    tfds.load( 'cats_vs_dogs',
               split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
               with_info=True,
               as_supervised=True )

print( raw_train )  # type: PrefetchDataset
print( raw_validation )
print( raw_test )
get_label_name = metadata.features[ 'label' ].int2str

for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow( image )
    plt.title( get_label_name(label) )

###################################
# Data Prepocessing
#
# raw_train -> train -> train_batch
#        (resize) (shuffle,batch)
###################################
print( image )
print( image.shape )

IMG_SIZE = 160 # image 160*160

def format_example( image, label ): # 이미지 값을 [-1,1] 으로
    image = tf.cast( image, tf.float32 )
    image = (image/127.5) -1
    image = tf.image.resize( image, (IMG_SIZE, IMG_SIZE))
    return image, label

# map() 함수는 built-in 함수로 list 나 dictionary 와 같은 iterable 한 데이터를, 함수를 인자로 전달하여 결과를 list로 형태로 반환해 주는 함수이다.
train = raw_train.map( format_example )           # raw_train ->(format_example)-> train
validation = raw_validation.map( format_example ) # type(raw_validation->validation): PrefetchDataset -> MapDataset
test = raw_test.map( format_example )

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

# shuffle( 1000 ) : 한번 epoch이 돌고나서 랜덤하게 섞을 것인지 정한다.
# batch( batch_size ) : batch size 를 정의
# type 변화: MapDataset -> BatchDataset
train_batches = train.shuffle( SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE) # batch 사이즈 정의 및 epoch이후 shuffle
validation_batches = validation.batch(BATCH_SIZE) # batch 사이즈 정의
test_batches = test.batch(BATCH_SIZE) # batch 사이즈 정의

for image_batch, label_batch in train_batches.take(1):
    pass
image_batch.shape # TensorShape([32, 160, 160, 3])


###################################
# Transfer Learning
#
# Google MobileNet V2
###################################

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
base_model = tf.keras.applications.MobileNetV2( input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet' )

feature_batch = base_model( image_batch )
print( feature_batch.shape ) # (32, 5, 5, 1280)

base_model.trainable = False
base_model.summary()

# average layer
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# dense layer
prediction_layer = tf.keras.layers.Dense(1)
# batch
feature_batch_average = global_average_layer( feature_batch )
print( feature_batch_average.shape )
prediction_batch = prediction_layer( feature_batch_average )
print( prediction_batch.shape )

# model
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])
# compile
base_learning_rate = 0.0001
model.compile( optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
               metrics=['accuracy'] )
model.summary()
len( model.trainable_variables )

# training 전 성능
initial_epochs = 5
validation_steps = 20

loss0, accuracy0 = model.evaluate( validation_batches, steps=validation_steps )
print("initial loss: {:.2f}".format( loss0 ))
print("initial accuracy: {:.2f}".format( accuracy0 ))

# training
history = model.fit( train_batches,
                     epochs=initial_epochs,
                     validation_data=validation_batches )


###################################
# Learning Curve
###################################

acc = history.history['accuracy']
loss = history.history['loss']

val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

plt.figure( figsize=(8,8) )
plt.subplot( 2, 1, 1 )
plt.plot( acc, label='trainig')
plt.plot( val_acc, label='validation')
plt.legend( loc='lower right' )
plt.ylabel('Accuracy')
plt.ylim( [min(plt.ylim()), 1] )
plt.title( 'Training and Validation Accuracy')

plt.subplot( 2, 1, 2 )
plt.plot( loss, label='trainig')
plt.plot( val_loss, label='validation')
plt.legend( loc='upper right' )
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title( 'Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


##########################################
# Transfer Learning / Fine-Tuning
##########################################

base_model.trainable = True

# Fine-Tuning 은 Layer의 뒷부분만 수행함
print( "Number of Layer:", len(base_model.layers) )
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]: # <- 앞부분의 layer는 lock 시킴
    layer.trainable = False

model.compile( optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),  # <- fine tune의 lr 는 1/10 했음
               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # digit+logic 같음. 양수1,음수0
               metrics=['accuracy'] )
model.summary( )

len( model.trainable_variables )

fine_turn_epochs = 5
total_epochs = initial_epochs + fine_turn_epochs # epochs 는 합해서

history_fine = model.fit( train_batches,
                          epochs=total_epochs,
                          initial_epoch=history.epoch[-1],  # <- fine tuning 은 이전에서 시작
                          validation_data=validation_batches )

acc += history_fine.history[ 'accuracy' ]
val_acc += history_fine.history[ 'val_accuracy' ]

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training')
plt.plot(val_acc, label='Validation')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training')
plt.plot(val_loss, label='Validation')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

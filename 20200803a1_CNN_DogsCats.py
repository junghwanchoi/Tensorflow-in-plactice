



import tensorflow as tf
import numpy as np


#############################################
# 파일 다운로드
#############################################
import os
import zipfile


SrcUrl = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"



# 방법#1
#저장위치: C:\Users\jungh\.keras\datasets\horse-or-human.zip
#path_to_zip = tf.keras.utils.get_file( 'horse-or-human.zip', origin=SrcUrl, extract=True)
#PATH        = os.path.join( os.path.dirname(path_to_zip), 'horse-or-human')

# 방법#2
local_zip = tf.keras.utils.get_file( 'cats_and_dogs_filtered.zip', origin=SrcUrl, extract=False) #저장위치: C:\Users\jungh\.keras\datasets\horse-or-human.zip
zip_ref   = zipfile.ZipFile( local_zip, 'r')
zip_ref.extractall('/tmp') # D:\tmp\horse-or-human
zip_ref.close()

# Directory path
train_cats_dir = os.path.join('/tmp/cats_and_dogs_filtered/train/cats') #경로를 병합하여 새 경로 생성
train_dogs_dir = os.path.join('/tmp/cats_and_dogs_filtered/train/dogs') #경로를 병합하여 새 경로 생성
# file name
train_cats_names = os.listdir( train_cats_dir )
train_dogs_names = os.listdir( train_dogs_dir )
print( "train_cats_names[:10]:", train_cats_names[:10])
print( "train_dogs_names[:10]:", train_dogs_names[:10])
print( "len(train_cats_names):", len(train_cats_names) )
print( "len(train_dogs_names):", len(train_dogs_names) )
print( "type(train_cats_names[0]):", type(train_cats_names[0]) )


###############################
# 데이터 확인: 4 * 4 이미지 표시
###############################

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

nrows = 4
ncols = 4

pic_index = 0
fig = plt.gcf() # get the current figure
fig.set_size_inches( ncols*4, nrows*4 )

pic_index += 8
next_cats_pix = [ os.path.join(train_cats_dir, fname) for fname in train_cats_names[pic_index-8:pic_index] ]
next_dogs_pix = [ os.path.join(train_dogs_dir, fname) for fname in train_dogs_names[pic_index-8:pic_index] ]

for i, img_path in enumerate( next_cats_pix+next_dogs_pix ): # i = 0, 1, ... (16-1)
    # Set up subplot
    sp = plt.subplot( nrows, ncols, i+1 ) # sp: subplot 1,2,...,16
    sp.axis( 'off' )

    img = mpimg.imread( img_path )
    plt.imshow( img )

plt.show( )


###############################
# Data Processing
###############################

from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

img_path = os.path.join(train_cats_dir, train_cats_names[0])
img = load_img( img_path )  # this is a PIL image
img_arry = img_to_array(img)  # Numpy array with shape (150, 150, 3)
print( train_cats_names[0] )
print( "leng:", len(img_arry) )


from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# train_datagen = ImageDataGenerator( rescale=1.0/255.0 )

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)



validation_datagen = ImageDataGenerator( rescale=1.0/255.0 )

train_generator = train_datagen.flow_from_directory(
    '/tmp/cats_and_dogs_filtered/train',
    target_size=(150,150),
    batch_size=20,  # 배치 크기를 지정합니다.
    class_mode='binary' # 2개 구분시:'binary', 다수 구분시:'categorical'
	                    # categorical : 2D one-hot 부호화된 라벨이 반환됩니다. 
                        # binary : 1D 이진 라벨이 반환됩니다.
                        # sparse : 1D 정수 라벨이 반환됩니다.
                        # None : 라벨이 반환되지 않습니다.
)

validation_generator = validation_datagen.flow_from_directory(
    '/tmp/cats_and_dogs_filtered/validation',
    target_size=(150,150),
    batch_size=20,  # 배치 크기를 지정합니다.
    class_mode='binary' # 2개 구분시:'binary', 다수 구분시:'categorical'
	                    # categorical : 2D one-hot 부호화된 라벨이 반환됩니다.
                        # binary : 1D 이진 라벨이 반환됩니다.
                        # sparse : 1D 정수 라벨이 반환됩니다.
                        # None : 라벨이 반환되지 않습니다.
)

###############################
# CNN
###############################

from tensorflow.keras.optimizers import RMSprop # binary classification

model = tf.keras.Sequential( [
    tf.keras.layers.Conv2D( 32, (3,3), activation='relu', input_shape=(150,150,3)), # 16은 채널수
    tf.keras.layers.MaxPool2D( 2,2 ),
    tf.keras.layers.Conv2D( 64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D( 2,2 ),
    tf.keras.layers.Conv2D( 128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D( 128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten( ),
    tf.keras.layers.Dense( 512, activation='relu' ), 
    tf.keras.layers.Dense( 1, activation='sigmoid' ) # 2개 구분시:'sigmoid', 다수 구분시:'softmax'
])

model.summary( )
model.compile( loss='binary_crossentropy', # 2개 구분시:'binary_crossentropy', 다수 구분시:'categorical_crossentropy'
               optimizer=RMSprop(lr=0.001),
               metrics=['accuracy'] )



###############################
# Training
###############################

history = model.fit(
    train_generator,
    steps_per_epoch=100, # 2000(샘플수) = 20(batch) * 100(step)
    epochs=15, # 전체 훈련 데이터셋에 대해 학습 반복 횟수를 지정합니다. 15번을 반복적으로 학습시켜 보겠습니다.
    verbose=2, # 진행상황을 ProgressBar(1)로 표시
    validation_data=validation_generator,
    validation_steps=50  # 1000(샘플수) = 20(batch) * 50(step)
)

model.save("cats_and_dogs_filtered.h5")

###############################
# Learning Curve
###############################

acc     = history.history[ 'accuracy' ]
val_acc = history.history[ 'val_accuracy' ]
loss    = history.history[ 'loss' ]
val_loss= history.history[ 'val_loss' ]
epochs  = range( len(acc) )

plt.figure( )
plt.plot( epochs, acc, 'b', label='training' )
plt.plot( epochs, val_acc, 'r', label='validation' )
plt.title( 'training and validation accuracy' )


plt.figure( )
plt.plot( epochs, loss, 'b', label='training' )
plt.plot( epochs, val_loss, 'r', label='validation' )
plt.title( 'training and validation loss' )


###############################
# Prediction using the Model
###############################

import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

cats_img_files = [os.path.join(train_cats_dir, f) for f in train_cats_names]
dogs_img_files = [os.path.join(train_dogs_dir, f) for f in train_dogs_names]
img_path = random.choice(cats_img_files + dogs_img_files) # 무작위 선택

# predicting images
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])

if classes[0]>0.5:
    print(img_path + " is a cat")
else:
    print(img_path + " is a dog")


plt.figure( )
plt.imshow( mpimg.imread(img_path) ) # *.png 파일 그리기
plt.axis('off')
plt.show()






###################################################
# Visualizing Intermediate Representations
###################################################

import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model.layers[1:]]
#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
# Let's prepare a random input image from the training set.
cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cats_names]
dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dogs_names]
img_path = random.choice(cat_img_files + dog_img_files)

img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
"""
    plt.figure(figsize=(scale * n_features, scale))

    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
plt.show()
"""


###################################################
# Clean Up
###################################################

import os, signal
os.kill( os.getpid(), signal.SIGTERM )
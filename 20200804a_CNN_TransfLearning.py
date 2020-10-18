
import os

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3

SrcUrl = 'https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# 사전 훈련 된 모델에서 기능을 추출하고 내 모델에서 사용
local_weights_file = tf.keras.utils.get_file( 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', origin=SrcUrl, extract=False)

pre_trained_model = InceptionV3( input_shape=(150,150,3),
                                 include_top=False,
                                 weights=None )

pre_trained_model.load_weights( local_weights_file )

for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

last_layer = pre_trained_model.get_layer( 'mixed7')
print( 'last layer output shape:', last_layer.output_shape )
last_output = last_layer.output


from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop

x = layers.Flatten()(last_output)                    # layer 쌓을때 lat_output > Flatten() > Dense 경우
x = layers.Dense( 1024, activation='relu' )(x)       #   여러줄  Layer0 = layers.Flatten( )
                                                     #          Layer1 = layers.Dense( 1024 )
                                                     #
                                                     #   한줄    Layer1 = Layers.Dense( 1024 )(Layer0)
x = layers.Dropout( 0.2 )(x) # 20%를 Dropout 시킴
x = layers.Dense( 1, activation='sigmoid' )(x)

model = Model( pre_trained_model.input, x )

model.compile( optimizer=RMSprop(lr=0.0001),
               loss = 'binary_crossentropy',
               metrics=['accuracy'] )

model.summary()








######################################################
# 다른 예제 (인터넷)
# 출처: https://3months.tistory.com/150 [Deep Play]
# 실행 안됨
######################################################


# Weight를 h5 파일 포맷으로 만들어 저장하기
model.save_weights("model.h5")
print("Saved model to disk")


# 로드한 모델에 Weight 로드하기
model.load_weights("model.h5")
print("Loaded model from disk")



# 모델 컴파일 후 Evaluation

model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])

# model evaluation
score = model.evaluate(X,Y,verbose=0)
print("%s : %.2f%%" % (model.metrics_names[1], score[1]*100))



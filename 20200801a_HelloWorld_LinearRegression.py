
import tensorflow as tf
import numpy as np


layer_0 = tf.keras.layers.Dense( units=1, input_shape=[1])
model = tf.keras.Sequential( [layer_0] )

model.compile( optimizer = 'sgd', loss="mean_squared_error", metrics=['accuracy'] )

xs = np.array( [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array( [-3.0,-1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit( xs, ys, epochs=500 )

print( model.predict( [5,0] ) )
print("Weight: {}".format(layer_0.get_weights()))

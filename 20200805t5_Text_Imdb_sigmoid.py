

#
# Multiple Layer GRU
#



from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
print(tf.__version__)




#############################################
# 파일 다운로드
#############################################
import tensorflow_datasets as tfds

# Get the data
dataset, dataset_info = tfds.load("imdb_reviews/subwords8k", 
                                  with_info=True, 
                                  as_supervised=True)

# WARNING:absl:TFDS datasets with text encoding are deprecated(더 이상 사용되지 않음) and will be removed in a future version. 
# Instead, you should use the plain text version and tokenize the text using `tensorflow_text` 
# (See: https://www.tensorflow.org/tutorials/tensorflow_text/intro#tfdata_example)


train_dataset, test_dataset = dataset['train'], dataset['test']

					   


#############################################
# Data Processing
#############################################


tokenizer = dataset_info.features['text'].encoder

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, train_dataset.output_shapes)
test_dataset  = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)



###############################
# Model : Embedding
###############################

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()




###############################
# Training
###############################

num_epochs = 10
history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
model.save("imdb_reviews.h5")




###############################
# Learning Curve
###############################

import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')



###############################
# Visualize the embeddings
###############################


# retrieve(재확인) the learned embeddings
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)


# vecs.tsv / meta.tsv 생성하기
# Embedding Projector (http://projector.tensorflow.org) 에서 사용가능 
# D:\Work\PyCharm\pythonProject1 에 vecs.tsv / meta.tsv 생성됨
import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, tokenizer.vocab_size):
  word = tokenizer.decode([word_num])
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

# 아래는 google colab 에서 동작하는 download기능 
'''
try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')
'''


###############################
# Prediction using the Model
###############################




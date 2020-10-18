
import tensorflow as tf



#############################################
# 파일 다운로드
#############################################
import tensorflow_datasets as tfds
import numpy as np

# TFDS 사용시, Parameter 및 Return값
imdb, info = tfds.load( 'imdb_reviews',
                        with_info=True,
                        as_supervised=True )



train_data, test_data = imdb['train'], imdb['test']

train_sentences = []
train_labels = []

test_sentences = []
test_labels = []

for s, l in train_data:
    train_sentences.append( str(s.numpy()) )
    train_labels.append( l.numpy() )

for s, l in test_data:
    test_sentences.append( str(s.numpy()) )
    test_labels.append( l.numpy() )

train_labels_final = np.array( train_labels )
test_labels_final = np.array( test_labels )


#############################################
# Data Processing
#############################################

vocab_size = 10000
embedding_dim = 16
max_length = 120
padding_type = 'post'
oov_tok = '<OOV>'

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

print("--- After Tokenizer : train, test ---")
print(len(train_sequences[0]))
print(len(train_sequences[1]))
print(len(train_sequences[10]))
print(len(train_padded[0]))
print(len(train_padded[1]))
print(len(train_padded[10]))
print(train_padded.shape)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, padding=padding_type, maxlen=max_length)
print(len(test_sequences))
print(test_padded.shape)

# word_index      reverse_word_index
# hello:1      ->    1:hello
reverse_word_index = dict( [(value, key) for (key, value) in word_index.items() ])

def decode_review( text ):
    return ' '.join( [reverse_word_index.get(i, '?') for i in text] )

print( train_sentences[3])
print( decode_review(train_padded[3]))




###############################
# Model : Embedding
###############################

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length), # Embedding( input_dim, output_dim, ...
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # binary:"sigmoid", multi-classification:"softmax"
])
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model.summary()

###############################
# Training
###############################

num_epochs = 10
history = model.fit(train_padded, train_labels_final, epochs=num_epochs, validation_data=(test_padded, test_labels_final))
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





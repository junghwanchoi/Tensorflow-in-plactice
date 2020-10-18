

import json
import tensorflow as tf
import csv
import random
import numpy as np




embedding_dim = 100
max_length = 16
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size=160000
test_portion=0.1



#############################################
# 파일 다운로드
#############################################

# 원 파일은 "Sentiment140 dataset with 1.6 million tweets"
# https://www.kaggle.com/kazanova/sentiment140
#
# This is the sentiment140 dataset.
# It contains 1,600,000 tweets extracted using the twitter api .
# The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment .

SrcUrl = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/training_cleaned.csv"
filepath = tf.keras.utils.get_file( "training_cleaned.csv", SrcUrl )





					   


#############################################
# Data Processing
#############################################
num_sentences = 0

corpus = []
with open(filepath, 'r', encoding='UTF8') as csvfile: #알고보니 파이썬3 부터는 ANSI 기준으로 작성된 파일만 정상적으로 읽어 올 수 있으며 UTF-8로 작성된 파일은 그냥 옆 코드로 읽으면 못 읽어 온다고 합니다.
    reader = csv.reader( csvfile, delimiter=',')
    for row in reader:
        list_item=[]
        list_item.append(row[5])
        this_label=row[0]
        if this_label=='0':
            list_item.append(0)
        else:
            list_item.append(1)
        num_sentences = num_sentences+1
        corpus.append(list_item)


print( num_sentences )
print( len(corpus) )
print( corpus[1] )

sentences=[]
labels=[]
random.shuffle(corpus)

for x in range( training_size ):
    sentences.append( corpus[x][0] )
    labels.append( corpus[x][1] )


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer( )
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
vocab_size = len( word_index )

sequences = tokenizer.texts_to_sequences( sentences )
padded = pad_sequences( sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

split = int( test_portion*training_size )

train_sequences = sequences[0:split]
test_sequences = sequences[split:training_size]
train_labels = labels[0:split]
test_labels = labels[split:training_size]

print( vocab_size )
print( word_index['i'] )


###############################
# Model : Embedding
###############################

# Note this is the 100 dimension version of GloVe from Stanford
# I unzipped and hosted it on my site to make this notebook easier
SrcUrl = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt"
filepath = tf.keras.utils.get_file( "glove.6B.100d.txt", SrcUrl )

# 파일에서 모든 글자의 학습값을 가져옴
embeddings_index = {} # dictionary
with open(filepath, 'r', encoding='UTF8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray( values[1:], dtype='float32' ) # save as numpy array
        embeddings_index[word] = coefs  # save as dictionary

# word_index에 해당하는 index만 가져옴
embeddings_matrix = np.zeros((vocab_size+1, embedding_dim))  # numpy
for word, i in word_index.items(): # dictionary word:index
    embedding_vector = embeddings_index.get(word) # dictionary "embedding_index" 의 key(word)로 value를 찾음. embedding_vector는 numpy array
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector  # dictionary embeddeings_index -> numpy embeddings_matrix

print(len(embeddings_matrix))
# Expected Output
# 138859

# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()


###############################
# Training
###############################
import numpy as np

# 구글 collab에는 실행되나
# 로컬 컴퓨터에서는 실행되지 않음
# 이유를 모름

print(type(train_sequences),type(train_labels))

train_sequences1 = np.array( train_sequences, dtype=np.float)
train_labels1 = np.array( train_labels, dtype=np.float)
test_sequences1 = np.array( test_sequences, dtype=np.float)
test_labels1 = np.array( test_labels, dtype=np.float)

print(type(train_sequences1),type(train_labels1))

num_epochs = 50
history = model.fit( train_sequences, train_labels1, epochs=num_epochs, validation_data=(test_sequences, test_labels1), verbose=2)

print("Training Complete")





###############################
# Learning Curve
###############################

# Learning Curve를 그려보면 그렇게 좋아 지지 않음

import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")




# Expected Output
# A chart where the validation loss does not increase sharply!


###############################
# Visualize the embeddings
###############################



###############################
# Prediction using the Model
###############################




# Sequence Models and Literature

import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np



#############################################
# 파일 다운로드
#############################################


# lyrics corpus
# !wget --no-check-certificate \
#     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt \
#     -O /tmp/irish-lyrics-eof.txt
# data = open('/tmp/irish-lyrics-eof.txt').read()
# test corpus

SrcUrl = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt"
filepath = tf.keras.utils.get_file( "irish-lyrics-eof.txt", SrcUrl )

data = open( filepath ).read( )

#############################################
# Data Processing
#############################################


corpus = data.lower().split("\n") # 구분자 '\n' 로 분리

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print( tokenizer.word_index)  # {'and': 1, 'the': 2, 'a': 3, 'in': 4, 'all': 5, 'i': 6, 'for': 7, 'of': 8, 'lanigans': 9, 'ball': 10, 'were': 11, 'at': 12, 'to': 13, .....}
print(total_words)  # 263


# n-gram 처리 ( 1,2 1,2,3 1,2,3,5 1,2,3,5,6 ... )
input_sequences = []
for line in corpus: # 한줄의 sentence 단위로 처리
    token_list = tokenizer.texts_to_sequences([line])[0]  # [4, 2, 66, 8, 67, 68, 69, 70]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

print(input_sequences)  # [[4, 2], [4, 2, 66], [4, 2, 66, 8], [4, 2, 66, 8, 67],....]

# pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]  # last word as label, the other words are input
print(xs[0], labels[0])  # [0 0 0 0 0 0 0 0 0 4] 2
print(xs[1], labels[1])  # [0 0 0 0 0 0 0 0 4 2] 66

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)  # transfer labels into one-hot vector

print(xs.shape, ys.shape)  # (453, 10) (453, 263)

print(tokenizer.word_index['in'])  # 4
print(tokenizer.word_index['the'])  # 2
print(tokenizer.word_index['town'])  # 66
print(tokenizer.word_index['of'])  # 8
print(tokenizer.word_index['athy'])  # 67
print(tokenizer.word_index['one'])  # 68
print(tokenizer.word_index['jeremy'])  # 69
print(tokenizer.word_index['lanigan'])  # 70

print(xs[6])
print(ys[6])
# [ 0  0  0  4  2 66  8 67 68 69]
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

print(xs[5])
print(ys[5])
# [ 0  0  0  0  4  2 66  8 67 68]
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

print(tokenizer.word_index)  # {'and': 1, 'the': 2, 'a': 3, 'in': 4, 'all': 5, 'i': 6, 'for': 7, 'of': 8, 'lanigans': 9, 'ball': 10, ...}



###############################
# Model : Embedding
###############################

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))
adam = Adam( lr=0.01 )
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

###############################
# Training
###############################

history = model.fit(xs, ys, epochs=100, verbose=1)

# if use lyrics corpus,
# Embeding output might be 100 instead of 64
# LSTM hidden units might be 150 instead of 20
# Epochs might be 500 instead of 100

model.summary()
# print( model )

###############################
# Learning Curve
###############################

import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()


plot_graphs(history, 'accuracy')


###############################
# Prediction using the Model
###############################


seed_text = "I've got a bad feeling about this"
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)
# Laurence went to dublin girls they were they round all were saw gave til hall ned your glisten made relations glisten might ask glisten call old daughter hall hall glisten glisten glisten cakes farm steps steps steps new went me me a call call call call replied til mchugh mchugh ned farm til hall ned ned ask glisten was mchugh ned ned terrible hall minute glisten glisten glisten cakes farm til hall hall ned farm steps steps lanigans ball ball were ball odaly tipped were odaly gave til ned ned glisten boys boys all were hearty hearty wall hearty nelly ned farm steps steps

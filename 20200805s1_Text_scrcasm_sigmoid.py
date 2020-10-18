

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json


vocab_size = 10000 # 단어 수
embedding_dim = 16 # embedding matrix의 feature dimension 크기
max_length = 100   # pad 되는 최대 크기
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
train_size = 20000 # 입력 문장 중에서 train으로 사용할 문장 수 ( 20000 / 26709 )


#############################################
# 파일 다운로드
#############################################



SrcUrl = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json"
filepath = tf.keras.utils.get_file( "sarcasm.json", SrcUrl )
with open( filepath, 'r' ) as f:
    datastore = json.load(f)


sentences = []
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

print( len(sentences) )

train_sentences = sentences[0:train_size]
test_sentences = sentences[train_size:]
train_labels = labels[0:train_size]
test_labels = labels[train_size:]


###############################
# Data Processing
###############################

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

print(len(word_index))
print(word_index)


train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(train_padded[0])
print(train_padded.shape)


test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


# Need this block to get it to work with TensorFlow 2.x
# TensorFlow 2.x 에서는 아래 코드 필요 없음
#      변환전에도 numpy.ndarry 임           
import numpy as np
train_padded = np.array(train_padded)
train_labels = np.array(train_labels)
test_padded = np.array(test_padded)
test_labels = np.array(test_labels)


###############################
# Model : Embedding
###############################


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

###############################
# Training
###############################

num_epochs = 30
history = model.fit(train_padded, train_labels, epochs=num_epochs, validation_data=(test_padded, test_labels), verbose=2)

model.save("sarcasm.h5")

###############################
# Learning Curve
###############################

import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")


###############################
# Visualize the embeddings
###############################

print(train_padded[2])
print(train_sentences[2])
print(labels[2])

# "word":index -> index:"word"
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# word_index 는 { "word":index, ... } 의 dictionary 타입의 예약자 
# dic( ) 은 dictionary 타입의 생성자

def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text]) # ' '.join( list ) : list 사이에 ' '을 추가하면서 붙이기
# dictionary.get('key') 이면 key에 해당하는 value를 return 함
# dictionary.get('key', '디폴트값') 이면 key에 해당하는 value가 없으면 '디폴터값' return 함

print(train_sentences[2]) # mom starting to fear son's web series closest thing she will have to grandchild
print(train_padded[2])
# result: 
#[ 153  890    2  891 1445 2215  595 5650  221  133   36   45    2 8864
#    0    0    0    0    0    0    0    0    0    0    0    0    0    0
#    ...
#    0    0]
print(decode_sentence(train_padded[2])) # mom starting to fear son's web series closest thing she will have to grandchild ? ? ? ? ? ? ?  ...




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
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
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

sentence = ["granny starting to fear spiders in the garden might be real", "game of thrones season finale showing this sunday night"]
sequences = tokenizer.texts_to_sequences(sentence)
print( sequences )
# result: [[1, 890, 2, 891, 1, 5, 4, 2565, 380, 22, 178], [249, 3, 1, 247, 3385, 2933, 20, 1551, 259]]

padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print( padded )

print('is_sarcastic') # 냉소적 
print(model.predict(padded))
# result : [[8.0871737e-01] [2.7206391e-07]]


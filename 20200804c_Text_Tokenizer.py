import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


#############################################
# Tokenizer: fit_on_texts
#############################################
sentenses = [
    'i love my dog',
    'I, love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer( num_words=100, oov_token="<OOV>" ) # 단어 빈도에 따른 사용할 단어 개수의 최대값. 가장 빈번하게 사용되는 num_words개의 단어만 보존합니다
                                                          # OOV(Out Of Vocabulary, 인식못하는단어)를 추가함

tokenizer.fit_on_texts( sentenses ) # sentenses 를 input으로 fit 시킴

word_index = tokenizer.word_index # 숫자로 정리된 결과
print( word_index ) # 결과:{'<OOV>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}

sequences = tokenizer.texts_to_sequences(sentenses) # senstenses를 sequences 로 변환
print( sentenses ) # 결과: ['i love my dog', 'I, love my cat', 'You love my dog!', 'Do you think my dog is amazing?']

padded = pad_sequences( sequences, maxlen=5 ) # 5자 로 pad를 채움
print( padded )


#############################################
# Test
#############################################
# Try with words that the tokenizer wasn't fit to
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data) # fit_on_texts( test_data ) 이후는 texts로 표시됨
print("\nTest Sequence = ", test_seq) # 결과 : Test Sequence =  [[5, 1, 3, 2, 4], [2, 4, 1, 2, 1]]

padded = pad_sequences(test_seq, maxlen=10)
print("\nPadded Test Sequence: ")
print(padded)
# 결과
# Padded Test Sequence:
# [[0 0 0 0 0 5 1 3 2 4]
#  [0 0 0 0 0 2 4 1 2 1]]


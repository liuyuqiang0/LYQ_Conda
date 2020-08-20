import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

sentences=[
    'I love, my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer=Tokenizer(num_words=100,oov_token='OOV') # 创建令牌生成器的实例
tokenizer.fit_on_texts(sentences)
word_index=tokenizer.word_index
print(word_index)

sequences=tokenizer.texts_to_sequences(sentences)
padded_seq=pad_sequences(sequences)
print(sequences)
print(padded_seq)

test_data=[
    'I really love my dog',
    'my dog love my manatee'
]

test_seq=tokenizer.texts_to_sequences(test_data)
print(test_seq)
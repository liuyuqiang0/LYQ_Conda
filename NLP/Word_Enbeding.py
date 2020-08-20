import wget,json,os
import numpy as np
import tensorflow as tf
import  matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


import tensorflow_datasets as tfds

def IMDB_Reviews():
    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

    train_data, test_data = imdb['train'], imdb['test']

    training_sentences = []
    training_labels = []

    testing_sentences = []
    testing_labels = []

    # str(s.tonumpy()) is needed in Python3 instead of just s.numpy()
    for s, l in train_data:  # 以张量形式存储的数据
        training_sentences.append(s.numpy().decode('utf8')) # 变成字符串
        training_labels.append(l.numpy())

    for s, l in test_data:
        testing_sentences.append(s.numpy().decode('utf8'))
        testing_labels.append(l.numpy())

    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)

    vocab_size = 10000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    oov_tok = "<OOV>"

    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences,maxlen=max_length)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    model.summary()

def Sarcasm_Dataset():
    vocab_size = 1000
    embedding_dim = 16
    max_length = 50
    trunc_type = 'post'  # 末尾截断
    padding_type = 'post'  # 末尾填充0
    oov_tok = "<OOV>"
    training_size = 20000

    with open("./sarcasm.json", 'r') as f:
        datastore = json.load(f)

    sentences = []
    labels = []
    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)  # 实例化分词器
    tokenizer.fit_on_texts(training_sentences)

    word_index = tokenizer.word_index

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    training_padded = np.array(training_padded)  # 转换数据类型方便训练(列表 -> numpy )
    training_labels = np.array(training_labels)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()

    num_epochs = 30
    history = model.fit(training_padded, training_labels,
                        epochs=num_epochs,
                        validation_data=(testing_padded, testing_labels),
                        verbose=2)

    def plot_graphs(ax,history, string):
        ax.plot(history.history[string])
        ax.plot(history.history['val_' + string])
        ax.set_xlabel("Epochs")
        ax.set_ylabel(string)
        ax.legend([string, 'val_' + string])

    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(20,10))  # Create a figure and a set of subplots.
    plot_graphs(ax[0],history, "acc")
    # fig,ax=plt.subplot(1,2,2)
    plot_graphs(ax[1],history, "loss")

    plt.show()

def Pre_Tokenized():
    import tensorflow_datasets as tfds

    imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)
    train_data, test_data = imdb['train'], imdb['test']
    tokenizer = info.features['text'].encoder
    print(tokenizer.subwords)
    sample_string = 'TensorFlow, from basics to mastery'
    tokenized_string = tokenizer.encode(sample_string)
    print('Tokenized string is {}'.format(tokenized_string))

    original_string = tokenizer.decode(tokenized_string)
    print('The original string: {}'.format(original_string))
    for ts in tokenized_string:
        print('{} ----> {}'.format(ts, tokenizer.decode([ts])))

if __name__=='__main__':

    # Sarcasm_Dataset()
    Pre_Tokenized()





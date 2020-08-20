import wget,json,os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def Get_data(url):
    if 'sarcasm.json' not in os.listdir('.'):
        wget.download(url)
    with open("./sarcasm.json", 'r') as f:
        datastore = json.load(f)

    sentences = []
    labels = []
    urls = []
    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])
        urls.append(item['article_link'])

    return sentences,labels,urls

def Preprocess(tokenizer,sentences):

    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    print(len(word_index)) # 单词种数

    sequences = tokenizer.texts_to_sequences(sentences)
    padded_seq = pad_sequences(sequences, padding='post')
    print(padded_seq.shape)  # 句子条数，每条句子最大长度
    print(padded_seq[0])

    return padded_seq




tokenizer = Tokenizer(oov_token="<OOV>")
sentences,labels,urls=Get_data('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json')
tokenizer = Tokenizer(oov_token="<OOV>")
padded_seq=Preprocess(tokenizer,sentences)
print(sentences[2])
print(padded_seq[2])
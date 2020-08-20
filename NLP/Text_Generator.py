import wget,json,os
import numpy as np
import tensorflow as tf
import  matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()




# data="In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a pound. \nHis father died and made him a man again \n Left him a farm and ten acres of ground. \nHe gave a grand party for friends and relations \nWho didnt forget him when come to the wall, \nAnd if youll but listen Ill make your eyes glisten \nOf the rows and the ructions of Lanigans Ball. \nMyself to be sure got free invitation, \nFor all the nice girls and boys I might ask, \nAnd just in a minute both friends and relations \nWere dancing round merry as bees round a cask. \nJudy ODaly, that nice little milliner, \nShe tipped me a wink for to give her a call, \nAnd I soon arrived with Peggy McGilligan \nJust in time for Lanigans Ball. \nThere were lashings of punch and wine for the ladies, \nPotatoes and cakes; there was bacon and tea, \nThere were the Nolans, Dolans, OGradys \nCourting the girls and dancing away. \nSongs they went round as plenty as water, \nThe harp that once sounded in Taras old hall,\nSweet Nelly Gray and The Rat Catchers Daughter,\nAll singing together at Lanigans Ball. \nThey were doing all kinds of nonsensical polkas \nAll round the room in a whirligig. \nJulia and I, we banished their nonsense \nAnd tipped them the twist of a reel and a jig. \nAch mavrone, how the girls got all mad at me \nDanced til youd think the ceiling would fall. \nFor I spent three weeks at Brooks Academy \nLearning new steps for Lanigans Ball. \nThree long weeks I spent up in Dublin, \nThree long weeks to learn nothing at all,\n Three long weeks I spent up in Dublin, \nLearning new steps for Lanigans Ball. \nShe stepped out and I stepped in again, \nI stepped out and she stepped in again, \nShe stepped out and I stepped in again, \nLearning new steps for Lanigans Ball. \nBoys were all merry and the girls they were hearty \nAnd danced all around in couples and groups, \nTil an accident happened, young Terrance McCarthy \nPut his right leg through miss Finnertys hoops. \nPoor creature fainted and cried Meelia murther, \nCalled for her brothers and gathered them all. \nCarmody swore that hed go no further \nTil he had satisfaction at Lanigans Ball. \nIn the midst of the row miss Kerrigan fainted, \nHer cheeks at the same time as red as a rose. \nSome of the lads declared she was painted, \nShe took a small drop too much, I suppose. \nHer sweetheart, Ned Morgan, so powerful and able, \nWhen he saw his fair colleen stretched out by the wall, \nTore the left leg from under the table \nAnd smashed all the Chaneys at Lanigans Ball. \nBoys, oh boys, twas then there were runctions. \nMyself got a lick from big Phelim McHugh. \nI soon replied to his introduction \nAnd kicked up a terrible hullabaloo. \nOld Casey, the piper, was near being strangled. \nThey squeezed up his pipes, bellows, chanters and all. \nThe girls, in their ribbons, they got all entangled \nAnd that put an end to Lanigans Ball."
# corpus = data.lower().split("\n")  # 先转换成小写，然后按行分开

def Get_Data(url):
    if 'irish-lyrics-eof.txt' not in os.listdir('.'):
        wget.download(url)
    data=open('irish-lyrics-eof.txt').read()  # 连同换行符一起读入
    corpus = data.lower().split("\n")  # 先转换成小写，然后按行分开
    return corpus

corpus=Get_Data('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus) # 生成键值对以及语料库
total_words = len(tokenizer.word_index) + 1  # 添加一个外在的词汇


n_dims=100  # 设置一些超参数，超参数在训练时可以多次调试
Lstm_Unit=150
learnning_rate=0.01


input_sequences = []
max_sequence_len = 0
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]  # 每一行歌词转换为数值列表
    max_sequence_len=max(max_sequence_len,len(token_list))
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)   # 随着语料库文本的加大，所耗费的内存也是非常巨大的



input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))  # 前缀填充0

# create predictors and label
xs = input_sequences[:,:-1]
labels = input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)  # one-hot编码，创建标签的独热编码方便分类

model = Sequential()
model.add(Embedding(total_words, n_dims, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(Lstm_Unit)))  # 设置LSTM单元个数
model.add(Dense(total_words, activation='softmax'))
adam=Adam(lr=learnning_rate)  # 创建优化器并设置学习率
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
history = model.fit(xs, ys, epochs=100, verbose=2)

# plot_graphs(history, 'acc')

# seed_text = "Laurence went to dublin"

seed_text = "I've got a bad feeling about this"
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict_classes(token_list, verbose=0)  # Generate class predictions for the input samples
    print(predicted, type(predicted))
    output_word = tokenizer.index_word[int(predicted)]

    # for word, index in tokenizer.word_index.items():  # tokenizer.index_word[index]
    #     if index == predicted:
    #         output_word = word
    #         break
    seed_text += " " + output_word
print(seed_text)



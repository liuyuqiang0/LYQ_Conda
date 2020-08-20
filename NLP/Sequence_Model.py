import tensorflow as tf
# from tensorflow.keras

model=tf.keras.Sequential(
    tf.keras.layers.Embedding(tokenizer.vocab_size,64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),  # 64表示希望从此LSTM层输出的数量，双向会加倍
    # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64))
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
)
model.summary()
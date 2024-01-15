from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import sys
import codecs
import os
import json

nepochs = 40
batchsize = 256
lr = 0.01
maxlen = 24


f = open('./data/source.txt', 'r')
music_as_chunks = []
for elm in f:
  music_as_chunks.append(elm.rstrip("\n"))
f.close()

music_as_chunks = music_as_chunks[1:]

unique_chunks = sorted(list(set(music_as_chunks)))
chunk_indices = dict((c, i) for i, c in enumerate(unique_chunks))
indices_chunk = dict((i, c) for i, c in enumerate(unique_chunks))

step = 1
part_of_songs = []
next_chunks = []
start_ix = 0
for i in range(start_ix, len(music_as_chunks) - maxlen, step):
  part_of_songs.append(music_as_chunks[i: i + maxlen])
  next_chunks.append(music_as_chunks[i + maxlen])

X = np.zeros((len(part_of_songs), maxlen, len(unique_chunks)), dtype=np.bool)
y = np.zeros((len(part_of_songs), len(unique_chunks)), dtype=np.bool)
for i, part_of_song in enumerate(part_of_songs):
  for t, chunk in enumerate(part_of_song):
    X[i, t, chunk_indices[chunk]] = 1
  y[i, chunk_indices[next_chunks[i]]] = 1

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(256, input_shape=(maxlen, len(unique_chunks))))
model.add(tf.keras.layers.Dense(len(unique_chunks)))
model.add(tf.keras.layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr))

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience = 3)

model.fit(X, y, batch_size=batchsize, epochs=nepochs, callbacks=[callback])


if not os.path.exists("./model/"):
  os.mkdir("./model/")
model.save('./model/model.h5');
model.save_weights("./model/weights.h5")

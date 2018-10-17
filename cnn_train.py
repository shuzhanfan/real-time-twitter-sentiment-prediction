#!/usr/bin/env python

import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from wordcloud import WordCloud
from tqdm import tqdm
from time import time
import multiprocessing
from scipy.stats import hmean, norm
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import utils
from sklearn.preprocessing import scale
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.layers import Input, concatenate, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D, LSTM
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

df_clean = pd.read_csv("tweets_cleaned.csv", index_col=0)
df_clean.dropna(inplace=True)
df_clean.reset_index(drop=True, inplace=True)
X = df_clean.text
y = df_clean.target
X_train, X_vali_test, y_train, y_vali_test = train_test_split(X, y, test_size=0.02, random_state=8081)
X_vali, X_test, y_vali, y_test = train_test_split(X_vali_test, y_vali_test, test_size=0.5, random_state=8081)

vocab_w2v_matrix = joblib.load("vocab_w2v_matrix.pkl")
X_train_sequences = joblib.load("X_train_sequences.pkl")
X_vali_sequences = joblib.load("X_vali_sequences.pkl")

input      = Input(shape=(50,), dtype='int32')
embedding  = Embedding(100000, 200, weights=[vocab_w2v_matrix], input_length=50, trainable=True)(input)
conv1      = Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1)(embedding)
conv1_pool = GlobalMaxPooling1D()(conv1)
conv2      = Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu', strides=1)(embedding)
conv2_pool = GlobalMaxPooling1D()(conv2)
conv3      = Conv1D(filters=100, kernel_size=4, padding='valid', activation='relu', strides=1)(embedding)
conv3_pool = GlobalMaxPooling1D()(conv3)
conv123_concat = concatenate([conv1_pool, conv2_pool, conv3_pool], axis=1)
full       = Dense(256, activation='relu')(conv123_concat)
dropout    = Dropout(0.2)(full)
output     = Dense(1)(dropout)
output     = Activation('sigmoid')(output)
cnn_model      = Model(inputs=[input], outputs=[output])
adam       = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
cnn_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

time_start = time()

best_cnn_model_file = "best_CNN_model.{epoch:02d}-{val_acc:.5f}.hdf5"
checkpoint = ModelCheckpoint(best_cnn_model_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor="val_acc", patience=5, mode="max")
reduce_lr  = ReduceLROnPlateau(monitor="val_acc", factor=0.5, patience=2, min_lr=0.000001)
cnn_model.fit(X_train_sequences, y_train, batch_size=128, epochs=5, validation_data=(X_vali_sequences, y_vali), callbacks = [checkpoint, early_stop, reduce_lr])

time_end = time()

print "model running time: {0:.2f}s".format(time_end-time_start)


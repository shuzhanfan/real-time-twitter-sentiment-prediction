#!/usr/bin/env python

import os,sys
from datetime import datetime
import pytz
import re
from ConfigParser import SafeConfigParser
import logging
import tweepy
import json
from pymongo import MongoClient
import numpy as np
import pandas as pd
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dropout, Dense
from keras.layers.embeddings import Embedding
from keras.layers import Input, concatenate, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from helper_functions import _get_config, _logger

#This method will convert the 'created at' in GMT to CT (Central Time)
def gmt_to_ct(created_at):
    d_obj = datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')
    d_obj = d_obj.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('US/Central'))
    d_str = d_obj.strftime('%Y-%m-%d %H:%M:%S')
    return d_str

vocab_w2v_matrix = joblib.load("../vocab_w2v_matrix.pkl")
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
cnn_model.load_weights("../best_CNN_model.02-0.82600.hdf5")
adam       = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
cnn_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

html_pattern = r"https?://[^ ]+"
www_pattern = r"www.[^ ]+"
user_pattern = r"@[A-Za-z0-9_]+"
url_user_pattern = r"|".join((html_pattern, www_pattern, user_pattern))
negations = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
             "don't":"do not", "doesn't":"does not", "didn't":"did not",
             "haven't":"have not", "hasn't":"has not", "hadn't":"had not",
             "won't":"will not", "wouldn't":"would not",
             "can't":"can not", "couldn't":"could not",
             "shouldn't":"should not",
             "mightn't":"might not",
             "mustn't":"must not"}
negations_pattern = re.compile(r'\b(' + '|'.join(negations.keys()) + r')\b')
tokenizer = WordPunctTokenizer()

def clean_tweet(tweet):
    soup = BeautifulSoup(tweet, "lxml")
    souped = soup.get_text()
    url_clean = re.sub(url_user_pattern, "", souped)
    try:
        bom_clean = url_clean.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_clean = url_clean
    negation_clean = negations_pattern.sub(lambda x: negations[x.group()], bom_clean)
    letters_only = re.sub("[^a-zA-Z]", " ", negation_clean)
    lower = letters_only.lower()
    words = [x for x in tokenizer.tokenize(lower) if len(x) > 1]
    tweet_clean = (" ".join(words)).strip()
    return tweet_clean

df_clean = pd.read_csv("../tweets_cleaned.csv", index_col=0)
df_clean.dropna(inplace=True)
df_clean.reset_index(drop=True, inplace=True)
X = df_clean.text
y = df_clean.target
X_train, X_vali_test, y_train, y_vali_test = train_test_split(X, y, test_size=0.02, random_state=8081)
X_vali, X_test, y_vali, y_test = train_test_split(X_vali_test, y_vali_test, test_size=0.5, random_state=8081)
tokenizer_keras = joblib.load("../tokenizer_keras.pkl")

def predict_sentiment(tweet):
    '''
    This method predicts the sentiment value of a single tweet based on the best performed deep learning model.
    '''
    tweet_cleaned = clean_tweet(tweet)
    sequence_tweet = tokenizer_keras.texts_to_sequences([tweet_cleaned])
    sequence_tweet_padded = pad_sequences(sequence_tweet, maxlen=50)
    sentiment = cnn_model.predict(sequence_tweet_padded)[0][0]
    return sentiment

class MyStreamListener(tweepy.StreamListener):
    '''
    This class inherits from tweepy.StreamListener to connect to Twitter Streaming API.
    '''
    def on_connect(self):
        logger.info('......Connected to Twitter Streaming API...... \n')

    def on_data(self, raw_data):
        try:
            client = MongoClient(mongo_host)
            db = client.twitter_sentiment
            data = json.loads(raw_data)
            if not data['retweeted'] and 'RT @' not in data['text'] and not data['in_reply_to_status_id']:
                created_at = data['created_at']
                tweeted_at = gmt_to_ct(created_at)
                text = data['text']
                sentiment = predict_sentiment(text)
                lang = data['user']['lang']
                if text and lang == 'en':
                    data_mongo = {}
                    data_mongo['tweeted_at'] = tweeted_at
                    data_mongo['sentiment'] = str(sentiment)
                    data_mongo['insert_time'] = str(10000*(time.time()))
                    logger.info('Tweeted at %s Central Time....' % tweeted_at)
                    logger.info(text)
                    logger.info(sentiment)
                    logger.info("  ")
                    collection_name = "twitter_sentiment_" + topic
                    db_coll = db[collection_name]
                    db_coll.insert_one(data_mongo)
        except Exception as e:
            logger.error(e)

    def on_error(self, status_code):
        print status_code

if __name__ == "__main__":
    logger = _logger()
    parser = _get_config()
    consumer_key = parser.get('Keys', 'consumer_key')
    consumer_secret = parser.get('Keys', 'consumer_serect')
    access_key = parser.get('Keys', 'access_token')
    access_serect = parser.get('Keys', 'access_secret')
    mongo_host = parser.get('Mongodb', 'host')
    topic = parser.get('TOPIC', 'topic')

    while True:
        try:
            auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_key, access_serect)
            api = tweepy.API(wait_on_rate_limit_notify=True)
            listener = MyStreamListener(api=api)
            streamer = tweepy.Stream(auth=auth, listener=listener)
            logger.info('......Collecting tweets......')
            streamer.filter(track=[topic], stall_warnings=True)
        except Exception as e:
            print e
            time.sleep(5)

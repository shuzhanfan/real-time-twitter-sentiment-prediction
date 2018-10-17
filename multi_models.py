#!/usr/bin/env python

import numpy as np
import pandas as pd
import re
from time import time
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from wordcloud import WordCloud
from tqdm import tqdm
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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
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

X_train_vali_w2vs_concat_mean = joblib.load("X_train_vali_w2vs_concat_mean.pkl")
X_train_w2vs_concat_mean, X_vali_w2vs_concat_mean = X_train_vali_w2vs_concat_mean

logistic     = LogisticRegression()
decision_tree  = DecisionTreeClassifier()
svm_linear     = LinearSVC()
random_forest  = RandomForestClassifier()
gradient_boost = GradientBoostingClassifier()
ml_models = [logistic, decision_tree, svm_linear, random_forest, gradient_boost]
ml_names  = ["Logistic regression", "Decision tree", "Linear SVM", "Random forest", "Gradient Boosting"]
ml_models_names = zip(ml_models, ml_names)

def compare_models(model_list, xtrain, ytrain, xtest, ytest):
    for model, name in model_list:
        print "Validation results for {}".format(name)
        print model
        time_start = time()
        model_fit = model.fit(xtrain, ytrain)
        y_pred    = model_fit.predict(xtest)
        time_end = time()
        accuracy = accuracy_score(ytest, y_pred)
        print "accuracy: {0:.2f}%".format(accuracy*100)
        print "model running time: {0:.2f}s".format(time_end-time_start)
        print " "
compare_models(ml_models_names, X_train_w2vs_concat_mean, y_train, X_vali_w2vs_concat_mean, y_vali)

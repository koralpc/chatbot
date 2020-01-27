#Author : Koralp Catalsakal

#Import modules
import nltk
nltk.download('punkt')
import flask
import io

from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
import pandas as pd
#import tflearn
import json
import random
from pandas.io.json import json_normalize

from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Input,Embedding


MAX_SEQ_LENGTH = 10
global json_df,vocab,label_dict

#Tokenization of patterns
def tokenize(data,stemmer):
    flat_json_df = json_normalize(data['intents'])

    all_words = []
    for pattern in flat_json_df['patterns']:
        for pat in pattern:
            all_words.extend(nltk.word_tokenize(pat))

    all_words = [lancaster_stemmer.stem(w.lower()) for w in all_words]
    all_words = set(all_words)
    return all_words,flat_json_df

# Create vocabulary from the corpus
def create_vocabulary(corpus):
    vocabulary_dict = {}
    NB_OF_WORDS = len(corpus)
    for idx,word in enumerate(corpus):
        vocabulary_dict[word] = idx+2
    vocabulary_dict['unk'] = 1
    return vocabulary_dict

def numerate_text(vocabulary,text):
    numerized_text = []
    for t in text:
        output_vec = []
        token = nltk.word_tokenize(t)
        for inner_t in token:
            stemmed = lancaster_stemmer.stem(inner_t.lower())
            if stemmed in vocabulary.keys():
                output_vec.append(vocabulary[stemmed])
            else:
                output_vec.append(vocabulary['unk'])
        numerized_text.append(output_vec)
    return numerized_text

def numerate_string(vocabulary,str):
    numerized_str = []
    token = nltk.word_tokenize(str)
    for inner_t in token:
        stemmed = lancaster_stemmer.stem(inner_t.lower())
        if stemmed in vocabulary.keys():
            numerized_str.append(vocabulary[stemmed])
        else:
            numerized_str.append(vocabulary['unk'])
    numerized_str = [numerized_str]
    return numerized_str

def prepare_labels(dataframe):

    class_dict = {}
    #One-hot encode output
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(np.asarray(dataframe['tag']).reshape(-1,1))

    all_labels = []
    for i in dataframe.index:
        for l in range(len(dataframe.iloc[i]['patterns'])):
            all_labels.append(onehot_encoded[i])
        class_dict[np.argwhere(onehot_encoded[i] == 1)[0][0]] =  dataframe.iloc[i]['tag']

    return np.array(all_labels,dtype = 'float32'),class_dict


def trainChatNN(X,y,vocabulary_length):
    X_reshaped = X.reshape(X.shape[0],1,X.shape[1])

    input_dim = X_reshaped.shape[1:3]
    output_classes = y.shape[1]
    ##TRAINING NETWORK NOW##
    model = Sequential()

    model.add(Input(input_dim[1]))
    model.add(Embedding(vocabulary_length + 1,input_dim[1]//2,input_length = 1,mask_zero =  True))
    model.add(LSTM(input_dim[1]*3, activation = 'relu' , return_sequences = False))
    model.add(Dense(input_dim[1]*2, activation = 'relu'))
    model.add(Dense(output_classes,activation ='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3000)

    history = model.fit(X,y, epochs=3000,verbose = 1,callbacks = [early_stop])
    model.save('chatbot.h5')
    return model


def loadChatbot(filename):
    loaded_model = tf.keras.models.load_model(filename)
    return loaded_model


def predictStringInput(model,str):
        numerical = numerate_string(vocab,str)
        padded = pad_sequences(numerical, maxlen=MAX_SEQ_LENGTH)
        padded = padded.reshape(1,padded.size)
        prediction = model.predict(padded)
        predicted_class = np.argmax(prediction)
        predicted_tag = label_dict[predicted_class]
        print('Input is: {0} - Predicted tag: {1} - Confidence : {2:.2f}'.format(str,predicted_tag, np.max(prediction)))
        responses = json_df['responses'][json_df['tag'] == predicted_tag]
        total_responses = len(responses.iloc[0])
        randidx = np.random.randint(0,total_responses)
        print('Response is: {0}'.format(responses.iloc[0][randidx]))
        return responses.iloc[0][randidx]

def interactiveLoop(model,vocabulary,dataframe,label_dictionary):
    print('Hello there! Wanna say something to me ?')
    user_in = ''
    while(True and user_in != 'Exit'):
        user_in = input()
        predictStringInput(model,user_in,vocabulary,dataframe,label_dictionary)
"""
        numerical = numerate_string(vocabulary,user_in)
        padded = pad_sequences(numerical, maxlen=MAX_SEQ_LENGTH)
        prediction = model.predict(padded)
        predicted_class = np.argmax(prediction)
        predicted_tag = label_dictionary[predicted_class]
        responses = dataframe['responses'][dataframe['tag'] == predicted_tag]
        print(responses)
        total_responses = len(responses.iloc[0])
        randidx = np.random.randint(0,total_responses-1)
        print(responses.iloc[0][randidx])
"""

#Load data
with open('intents.json') as file:
        data = json.load(file)

#Tokenize all words and apply stemming operation
all_words,json_df = tokenize(data,lancaster_stemmer)

#Gather all tags
tags = sorted(json_df['tag'])

#Create vocabulary from stemmed words(Not an embedding, just a vocabulary!)
vocab = create_vocabulary(all_words)

#Convert character arrays to numerical values, w.r.t our vocabulary
numerized = json_df['patterns'].apply(lambda x : numerate_text(vocab,x))

#Create the target label dataset
labels,label_dict = prepare_labels(json_df)

#Pad sequences to MAX_SEQ_LENGTH
padded = []
for n in numerized:
    padded.extend(list(pad_sequences(n, maxlen=MAX_SEQ_LENGTH)))
padded = np.array(padded,dtype = 'float32')

chatbot = trainChatNN(padded,labels,len(vocab))

user_in = input()
predictStringInput(chatbot,user_in)
#interactiveLoop(chatbot,vocab,json_df,label_dict)

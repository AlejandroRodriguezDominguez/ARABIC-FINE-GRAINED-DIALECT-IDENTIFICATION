# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:22:30 2020

@author: arodriguez
"""
from __future__ import absolute_import, division, print_function
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils import normalize
from camel_tools.utils import dediac
import os

import pandas as pd
import csv
import re
import nltk
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn import preprocessing

'''As we do not know Arabic Language, in order to help in the initial stages of the preprocessing stage (punctuations, special features of
arabic language) we use a section of the git library called motazsaad/process-arabic-text. Specifically the code called: clean_arabic_text.py
 https://github.com/motazsaad/process-arabic-text/blob/master/clean_arabic_text.py
 
 Functions include: normalize_arabic, remove_diacritics, remove_punctuations. remove_repeating_char '''
 

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
punctuations_list = arabic_punctuations 

arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
def unique(list1): 
  
    # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    
    return unique_list

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text


def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text


def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)




# STEP 1: DATA PREPROCESSING :  CLEANING DATA, REMOVE PUNCTUATION, REMOVE DIACRITICS,REMOVE REPEATED CHARS, NORMALIZE, AND TOKENIZE
    
'''We load data from MADAR Corpus 26 and 6 both for train and dev and we apply the previous preprocessing techniques plus Cammel Tools Tokenizer'''

filename1 = 'MADAR-Corpus-26-train.tsv'

train_26_X = []
train_26_Y = []
train_26_grams = []

with open(os.path.join('Project_1_tools_and_data/dataset/',filename1),encoding='utf8') as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for row in tsvreader:
        sentence1 = normalize_arabic(row[0])
        sentence2 = remove_diacritics(sentence1)
        sentence3 = remove_punctuations(sentence2)
        sentence4 = remove_repeating_char(sentence3)
        train_26_grams.append(sentence4)
        sentence = simple_word_tokenize(sentence4)
        train_26_X.append(sentence)
        train_26_Y.append(row[1])
        
filename2 = 'MADAR-Corpus-26-dev.tsv'
    
dev_26_X = []
dev_26_Y = []
dev_26_grams = []

with open(os.path.join('Project_1_tools_and_data/dataset/',filename2),encoding='utf8') as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for row in tsvreader:

        sentence1 = normalize_arabic(row[0])
        sentence2 = remove_diacritics(sentence1)
        sentence3 = remove_punctuations(sentence2)
        sentence4 = remove_repeating_char(sentence3)
        dev_26_grams.append(sentence4)
        sentence = simple_word_tokenize(sentence4)
        dev_26_X.append(sentence)
        dev_26_Y.append(row[1])    

 
filename3 = 'MADAR-Corpus-6-train.tsv'

train_6_X = []
train_6_Y = []
train_6_grams = []
with open(os.path.join('Project_1_tools_and_data/dataset/',filename3),encoding='utf8') as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for row in tsvreader:

        sentence1 = normalize_arabic(row[0])
        sentence2 = remove_diacritics(sentence1)
        sentence3 = remove_punctuations(sentence2)
        sentence4 = remove_repeating_char(sentence3)
        train_6_grams.append(sentence4)
        sentence = simple_word_tokenize(sentence4)
        train_6_X.append(sentence)
        train_6_Y.append(row[1])
        
filename4 = 'MADAR-Corpus-6-dev.tsv'
    
dev_6_X = []
dev_6_Y = []
dev_6_grams = []
with open(os.path.join('Project_1_tools_and_data/dataset/',filename4),encoding='utf8') as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for row in tsvreader:

        sentence1 = normalize_arabic(row[0])
        sentence2 = remove_diacritics(sentence1)
        sentence3 = remove_punctuations(sentence2)
        sentence4 = remove_repeating_char(sentence3)
        dev_6_grams.append(sentence4)
        sentence = simple_word_tokenize(sentence4)
        dev_6_X.append(sentence)
        dev_6_Y.append(row[1])   
        
        

 
 
# STEP 2: SYSTEM IMPLEMENTATION: 
        
'''IMPORTANT: In this section what we do is create all the data for the models and then the model is run only on one data at a time.
We did this because it was easier in order to make inference in the results by running the code one at a time with each case and seeing the results.
You only need to get the data from this section and plug it into the subsequent models'''

'''We have commented multiple cases so that if it is run it does not take long time. However, if it is uncommented it can be run all at once'''


# 1 - Feature-Based Classification for Dialectal Arabic (Data Processing)    

'''Word-gram features with uni-gram, bi-gram and tri-gram'''

'''Case document 26'''

'''Train Data'''
'''vectorizer1 = TfidfVectorizer(analyzer='word', ngram_range = (1,1), max_features = None, dtype=np.float32)'''
vectorizer2 = TfidfVectorizer(analyzer='word', ngram_range = (1,2), max_features = 50000, dtype=np.float32)
'''vectorizer3 = TfidfVectorizer(analyzer='word', ngram_range = (1,3), max_features = 20000, dtype=np.float32)'''


'''TFIDFNgram1_26 = vectorizer1.fit_transform(train_26_grams)'''
TFIDFNgram2_26 = vectorizer2.fit_transform(train_26_grams)
'''TFIDFNgram3_26 = vectorizer3.fit_transform(train_26_grams)'''


'''features1_26 = (vectorizer1.get_feature_names())'''
features2_26 = (vectorizer2.get_feature_names())
'''features3_26 = (vectorizer3.get_feature_names())'''


'''train_26_X_1 = TFIDFNgram1_26.toarray()'''
train_26_X_2 = TFIDFNgram2_26.toarray()
'''train_26_X_3 = TFIDFNgram3_26.toarray()'''

'''Dev Data'''

'''DFIDFNgram1_26 = vectorizer1.transform(dev_26_grams)'''
DFIDFNgram2_26 = vectorizer2.transform(dev_26_grams)
'''DFIDFNgram3_26 = vectorizer3.transform(dev_26_grams)'''

'''dev_26_X_1 = DFIDFNgram1_26.toarray()'''
dev_26_X_2 = DFIDFNgram2_26.toarray()
'''dev_26_X_3 = DFIDFNgram3_26.toarray()'''

'''Case document 6, gives memory error'''

'''vectorizer5 = TfidfVectorizer(ngram_range = (1,1), dtype=np.float32)
vectorizer6 = TfidfVectorizer(ngram_range = (2,2), dtype=np.float32)
vectorizer7 = TfidfVectorizer(ngram_range = (3,3), dtype=np.float32)


TFIDFNgram1_6 = vectorizer5.fit_transform(train_6_grams)
TFIDFNgram2_6 = vectorizer6.fit_transform(train_6_grams)
TFIDFNgram3_6 = vectorizer7.fit_transform(train_6_grams)


features1_6 = (vectorizer5.get_feature_names())
features2_6 = (vectorizer6.get_feature_names())
features3_6 = (vectorizer7.get_feature_names())


train_6_X_1 = TFIDFNgram1_6.toarray()
train_6_X_2 = TFIDFNgram2_6.toarray()
train_6_X_3 = TFIDFNgram3_6.toarray()'''



'''Character-gram features with/without word boundary consideration, from bi-gram and
up to 5-gram.'''

'''Case document 26'''

'''With Word Boundary Consideration: N-grams from characters inside word boundaries'''

'''Train Data'''
'''vectorizer2_cwb = TfidfVectorizer(analyzer='char_wb', ngram_range = (2,2), max_features = 20000, dtype=np.float32)
vectorizer3_cwb = TfidfVectorizer(analyzer='char_wb', ngram_range = (3,3), max_features = 20000, dtype=np.float32)
vectorizer4_cwb = TfidfVectorizer(analyzer='char_wb', ngram_range = (4,4), max_features = 20000, dtype=np.float32)
vectorizer5_cwb = TfidfVectorizer(analyzer='char_wb', ngram_range = (5,5), max_features = 20000, dtype=np.float32)
vectorizer_combined_cwb = TfidfVectorizer(analyzer='char_wb', ngram_range = (2,5), max_features = 20000, dtype=np.float32)


TFIDF_Cgram2_26_wb = vectorizer2_cwb.fit_transform(train_26_grams)
TFIDF_Cgram3_26_wb = vectorizer3_cwb.fit_transform(train_26_grams)
TFIDF_Cgram4_26_wb = vectorizer4_cwb.fit_transform(train_26_grams)
TFIDF_Cgram5_26_wb = vectorizer5_cwb.fit_transform(train_26_grams)
TFIDF_Cgram_combined_26_wb = vectorizer_combined_cwb.fit_transform(train_26_grams)


features2_C_26_wb = (vectorizer2_cwb.get_feature_names())
features3_C_26_wb = (vectorizer3_cwb.get_feature_names())
features4_C_26_wb = (vectorizer4_cwb.get_feature_names())
features5_C_26_wb = (vectorizer5_cwb.get_feature_names())
features_combined_C_26_wb = (vectorizer_combined_cwb.get_feature_names())

train_26_C_X_2_wb = TFIDF_Cgram2_26_wb.toarray()
train_26_C_X_3_wb = TFIDF_Cgram3_26_wb.toarray()
train_26_C_X_4_wb = TFIDF_Cgram4_26_wb.toarray()
train_26_C_X_5_wb = TFIDF_Cgram5_26_wb.toarray()
train_26_C_X_combined_wb = TFIDF_Cgram_combined_26_wb.toarray()'''

'''Dev Data'''

'''DFIDF_Cgram2_26_wb = vectorizer2_cwb.transform(dev_26_grams)
DFIDF_Cgram3_26_wb = vectorizer3_cwb.transform(dev_26_grams)
DFIDF_Cgram4_26_wb = vectorizer4_cwb.transform(dev_26_grams)
DFIDF_Cgram5_26_wb = vectorizer5_cwb.transform(dev_26_grams)
DFIDF_Cgram_combined_26_wb = vectorizer_combined_cwb.transform(dev_26_grams)

dev_26_C_X_2_wb = DFIDF_Cgram2_26_wb.toarray()
dev_26_C_X_3_wb = DFIDF_Cgram3_26_wb.toarray()
dev_26_C_X_4_wb = DFIDF_Cgram4_26_wb.toarray()
dev_26_C_X_5_wb = DFIDF_Cgram5_26_wb.toarray()
dev_26_C_X_combined_wb = DFIDF_Cgram_combined_26_wb.toarray()'''

'''Without Word Boundary Consideration'''

'''Train Data'''

'''vectorizer2_c = TfidfVectorizer(analyzer='char', ngram_range = (2,2), max_features = 20000, dtype=np.float32)
vectorizer3_c = TfidfVectorizer(analyzer='char', ngram_range = (3,3), max_features = 20000, dtype=np.float32)
vectorizer4_c = TfidfVectorizer(analyzer='char', ngram_range = (4,4), max_features = 20000, dtype=np.float32)
vectorizer5_c = TfidfVectorizer(analyzer='char', ngram_range = (5,5), max_features = 20000, dtype=np.float32)
vectorizer_combined_c = TfidfVectorizer(analyzer='char', ngram_range = (1,3), max_features = 20000, dtype=np.float32)


TFIDF_Cgram2_26 = vectorizer2_c.fit_transform(train_26_grams)
TFIDF_Cgram3_26 = vectorizer3_c.fit_transform(train_26_grams)
TFIDF_Cgram4_26 = vectorizer4_c.fit_transform(train_26_grams)
TFIDF_Cgram5_26 = vectorizer5_c.fit_transform(train_26_grams)
TFIDF_Cgram_combined_26 = vectorizer_combined_c.fit_transform(train_26_grams)


features2_C_26 = (vectorizer2_c.get_feature_names())
features3_C_26 = (vectorizer3_c.get_feature_names())
features4_C_26 = (vectorizer4_c.get_feature_names())
features5_C_26 = (vectorizer5_c.get_feature_names())
features_combined_C_26 = (vectorizer_combined_c.get_feature_names())

train_26_C_X_2 = TFIDF_Cgram2_26.toarray()
train_26_C_X_3 = TFIDF_Cgram3_26.toarray()
train_26_C_X_4 = TFIDF_Cgram4_26.toarray()
train_26_C_X_5 = TFIDF_Cgram5_26.toarray()
train_26_C_X_combined = TFIDF_Cgram_combined_26.toarray()'''



'''Dev Data'''

'''DFIDF_Cgram2_26 = vectorizer2_c.transform(dev_26_grams)
DFIDF_Cgram3_26 = vectorizer3_c.transform(dev_26_grams)
DFIDF_Cgram4_26 = vectorizer4_c.transform(dev_26_grams)
DFIDF_Cgram5_26 = vectorizer5_c.transform(dev_26_grams)
DFIDF_Cgram_combined_26 = vectorizer_combined_c.transform(dev_26_grams)

dev_26_C_X_2 = DFIDF_Cgram2_26.toarray()
dev_26_C_X_3 = DFIDF_Cgram3_26.toarray()
dev_26_C_X_4 = DFIDF_Cgram4_26.toarray()
dev_26_C_X_5 = DFIDF_Cgram5_26.toarray()
dev_26_C_X_combined = DFIDF_Cgram_combined_26.toarray()


le1 = preprocessing.LabelEncoder()
labels_train_1 = le1.fit_transform(train_26_Y)    
labels_dev_1 = le1.transform(dev_26_Y)   '''
    




# 1 - Feature-Based Classification for Dialectal Arabic (Modelling)

'''MODELLING'''

'''here we have commented the implementation of the Neural Net and leave it the Random Forest, but both were used during the assigment

in the fit section and predict you can put any case from the proprocessing stage above with their respective name. ie: train_26_X_2, train_26_X_3....''' 

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

'''NNclf = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam',
                      alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
                      power_t=0.5, max_iter=200, shuffle=True, random_state=0, tol=0.0001, verbose=False, 
                      warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, 
                      beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)'''

NNclf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=50, min_samples_split=2, 
                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                               max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, 
                               oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, 
                               ccp_alpha=0.0, max_samples=None)


NNclf.fit(train_26_X_2, train_26_Y)

pred_NN=NNclf.predict(dev_26_X_2)

acc = accuracy_score(dev_26_Y, pred_NN)
print(acc)

predicted_labels = pred_NN

'''Here we copy paste the section of the code for the MADAR_DID_SCORER.py because it was easier for us to run the code in this way instead
of running the script separately but the code and results are the same'''

gold_labels = dev_26_Y
labels = list(set(gold_labels))

accuracy = accuracy_score(gold_labels, predicted_labels) * 100 
f1 = f1_score(gold_labels, predicted_labels, labels = labels, average = None) * 100
recall = recall_score(gold_labels, predicted_labels, labels = labels, average = None) * 100
precision = precision_score(gold_labels, predicted_labels, labels = labels, average = None) * 100


print ("INDIVIDUAL PRECISION SCORE:")
for x in range (len(labels)):
    print (labels[x], ", PRECISION SCORE: %.2f" %precision[x], "%")
    
print ("\nINDIVIDUAL RECALL SCORE:")
for x in range (len(labels)):
    print (labels[x], ", RECALL SCORE: %.2f" %recall[x], "%")

print ("\nINDIVIDUAL F1 SCORE:")
for x in range (len(labels)):
    print (labels[x], ", F1 SCORE: %.2f" %f1[x], "%")



## computes overall scores (accuracy, f1, recall, precision)
accuracy = accuracy_score(gold_labels, predicted_labels) * 100
f1 = f1_score(gold_labels, predicted_labels, average = "macro") * 100
recall = recall_score(gold_labels, predicted_labels, average = "macro") * 100
precision = precision_score(gold_labels, predicted_labels, average = "macro") * 100


print ("\nOVERALL SCORES:")
## prints overall scores (accuracy, f1, recall, precision)
print ("MACRO AVERAGE PRECISION SCORE: %.2f" %precision, "%")
print ("MACRO AVERAGE RECALL SCORE: %.2f" %recall, "%")
print ("MACRO AVERAGE F1 SCORE: %.2f" %f1, "%")
print ("OVERALL ACCURACY: %.2f" %accuracy, "%\n")


'''labels_inverse = le.inverse_transform(labels)'''








# 2- LSTM Deep Network



import re
import numpy as np
from nltk import ngrams

'''this is a own made function for accuracy to complement the ones in MADAR_DIR'''

def calculate_accuracy(prediction,labels):
    indexpred=np.argmax(prediction,axis=1)
    counter=(indexpred == labels).sum()
    Accuracy=counter/prediction.shape[0]


    return Accuracy


'''----------MODELLING----------'''

import gensim
from nltk import ngrams
import spacy
import wget
from gensim.test.utils import *
import gensim.downloader
import zipfile
from gensim.models import Word2Vec
import tensorflow as tf


'''MAX_SEQUENCE_LENGTH = len(max(train_26_X, key=len))'''
MAX_SEQUENCE_LENGTH = 10
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2



'''this part of the code is from Ted Scully Deep Learning course at CIT, lecture on LSTM from the MSc on Artificial Intelligence
except the embedding section which is from gensim library. Here you put the embedding document to load the embeddings. It is necessary to have the document .mld
with other documents in the python script foulder as per the instructions on gensim library'''
 
t_model = gensim.models.Word2Vec.load('full_grams_cbow_100_twitter.mdl')

pretrained_weights = t_model.wv.syn0
liat = t_model.wv
word_vectors = t_model.wv.vocab

embeddings_index = {}
for i, word in enumerate(word_vectors):
    embeddings_index[word]=pretrained_weights[i]
    
print('Found %s word vectors.' % len(embeddings_index))

le = preprocessing.LabelEncoder()
labels_train = le.fit_transform(train_26_Y)    
labels_dev = le.transform(dev_26_Y)     

        
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(train_26_grams)
sequences_train = tokenizer.texts_to_sequences(train_26_grams)
sequences_dev = tokenizer.texts_to_sequences(dev_26_grams)

word_index = tokenizer.word_index
print('Found ',len(word_index),' unique tokens.')

        
# Pad the sequences so that each sequence is the same size
data_train = tf.keras.preprocessing.sequence.pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
data_dev = tf.keras.preprocessing.sequence.pad_sequences(sequences_dev, maxlen=MAX_SEQUENCE_LENGTH)

# This will one hot encode the numerical labels for each class. 
print('Shape of data train tensor:', data_train.shape)
print('Shape of data dev tensor:', data_dev.shape)
print('Shape of label train tensor:', labels_train.shape)
print('Shape of label dev tensor:', labels_dev.shape)

# split the data into a training set and a validation set
indices = np.arange(data_train.shape[0])
np.random.seed(0)
np.random.shuffle(indices)
data_train = data_train[indices]
labels_train = labels_train[indices]
num_validation_samples = int(VALIDATION_SPLIT * data_train.shape[0])

x_train = data_train[:-num_validation_samples]
y_train = labels_train[:-num_validation_samples]
x_val = data_train[-num_validation_samples:]
y_val = labels_train[-num_validation_samples:]


num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
print (num_words)

# The embedding matrix dimension will be the num of words in our vocabulary
# by the embedding dimensional
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

# iterate through every word and ID in the word/ int ID dictionary
for word, i in word_index.items():
  
  # all words in our vocabulary are encoded between and 0 and num_words-1
  # this just checks that i is a valid word index 
  if i < num_words:
    
    # if this word is contained in the downloaded embedding vector
    # then add it to our embedding matrix. 
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector
      
'''here we have commented some of the models we used and we leave one with no comments to be run'''

'''model = tf.keras.models.Sequential()
# The first layer is an embedded layer 
model.add(tf.keras.layers.Embedding(num_words, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, recurrent_dropout=0.3,return_sequences=True))) 
model.add(tf.keras.layers.Dropout(0.2,seed=0))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, recurrent_dropout=0.3)))
model.add(tf.keras.layers.Dense(150, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2,seed=0))
model.add(tf.keras.layers.Dense(26, activation='softmax'))
model.summary()'''

'''model = tf.keras.models.Sequential()
# The first layer is an embedded layer 
model.add(tf.keras.layers.Embedding(num_words, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(tf.keras.layers.LSTM(200, recurrent_dropout=0.3)) 
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(26, activation='softmax'))
model.summary()'''

model = tf.keras.models.Sequential()
# The first layer is an embedded layer 
model.add(tf.keras.layers.Embedding(num_words, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(tf.keras.layers.LSTM(300, recurrent_dropout=0.3,return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2,seed=0))
model.add(tf.keras.layers.LSTM(200, recurrent_dropout=0.3))
model.add(tf.keras.layers.Dense(150, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2,seed=0))
model.add(tf.keras.layers.Dense(26, activation='softmax'))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = True

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=15, batch_size=100, validation_data=(x_val, y_val))


predictions = model.predict(data_dev)

Accuracy=calculate_accuracy(predictions,labels_dev)
print('Model 2 Accuracy Test: \n',Accuracy)

predicted_labels = []
for i in range(predictions.shape[0]):
    indexpred=np.argmax(predictions[i])
    predicted_labels.append(indexpred)

'''We apply the same MADAR_DIR_Score.py section of the code in this part for the Evaluation section of the assigment'''


gold_labels = labels_dev
labels = list(set(gold_labels))

accuracy = accuracy_score(gold_labels, predicted_labels) * 100 
f1 = f1_score(gold_labels, predicted_labels, labels = labels, average = None) * 100
recall = recall_score(gold_labels, predicted_labels, labels = labels, average = None) * 100
precision = precision_score(gold_labels, predicted_labels, labels = labels, average = None) * 100


print ("INDIVIDUAL PRECISION SCORE:")
for x in range (len(labels)):
    print (labels[x], ", PRECISION SCORE: %.2f" %precision[x], "%")
    
print ("\nINDIVIDUAL RECALL SCORE:")
for x in range (len(labels)):
    print (labels[x], ", RECALL SCORE: %.2f" %recall[x], "%")

print ("\nINDIVIDUAL F1 SCORE:")
for x in range (len(labels)):
    print (labels[x], ", F1 SCORE: %.2f" %f1[x], "%")



## computes overall scores (accuracy, f1, recall, precision)
accuracy = accuracy_score(gold_labels, predicted_labels) * 100
f1 = f1_score(gold_labels, predicted_labels, average = "macro") * 100
recall = recall_score(gold_labels, predicted_labels, average = "macro") * 100
precision = precision_score(gold_labels, predicted_labels, average = "macro") * 100


print ("\nOVERALL SCORES:")
## prints overall scores (accuracy, f1, recall, precision)
print ("MACRO AVERAGE PRECISION SCORE: %.2f" %precision, "%")
print ("MACRO AVERAGE RECALL SCORE: %.2f" %recall, "%")
print ("MACRO AVERAGE F1 SCORE: %.2f" %f1, "%")
print ("OVERALL ACCURACY: %.2f" %accuracy, "%\n")


labels_inverse = le.inverse_transform(labels)










# 3 BERT Modelling

'''We use information from these two sources:
    - AraBERT: https://github.com/aub-mind/arabert
    - https://github.com/alisafaya/Arabic-BERT'''

'''@inproceedings{antoun2020arabert,
  title={AraBERT: Transformer-based Model for Arabic Language Understanding},
  author={Antoun, Wissam and Baly, Fady and Hajj, Hazem},
  booktitle={LREC 2020 Workshop Language Resources and Evaluation Conference 11--16 May 2020},
  pages={9}
}'''

'''@inproceedings{safaya-etal-2020-kuisail,
    title = "{KUISAIL} at {S}em{E}val-2020 Task 12: {BERT}-{CNN} for Offensive Speech Identification in Social Media",
    author = "Safaya, Ali  and
      Abdullatif, Moutasem  and
      Yuret, Deniz",
    booktitle = "Proceedings of the Fourteenth Workshop on Semantic Evaluation",
    month = dec,
    year = "2020",
    address = "Barcelona (online)",
    publisher = "International Committee for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.semeval-1.271",
    pages = "2054--2059",
}'''




''' For the code below to implement BERT we follow the same script as in the link from below:

https://towardsdatascience.com/multi-class-text-classification-with-deep-learning-using-bert-b59ca2f5c613'''




import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification


le1 = preprocessing.LabelEncoder()
labels_train_ = le1.fit_transform(train_26_Y)    
labels_dev_ = le1.transform(dev_26_Y)   



possible_labels = np.unique(labels_train_)

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
label_dict

train_val_split = 0.8

max_length = 20



tokenizer = BertTokenizer.from_pretrained('asafaya/bert-base-arabic', 
                                          do_lower_case=True,
                                          )
                                          
encoded_data_train = tokenizer.batch_encode_plus(
    train_26_grams[0:8000], 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=max_length, 
    return_tensors='pt'
)

encoded_data_dev = tokenizer.batch_encode_plus(
    dev_26_grams[0:3000], 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=max_length, 
    return_tensors='pt'
)







input_ids_train = encoded_data_train['input_ids']
input_ids_train = torch.tensor(input_ids_train, dtype=torch.long)
attention_masks_train = encoded_data_train['attention_mask']
attention_masks_train = torch.tensor(attention_masks_train, dtype=torch.long)
labels_train = torch.tensor(labels_train_, dtype=torch.long)
input_ids_dev = encoded_data_dev['input_ids']
input_ids_dev = torch.tensor(input_ids_dev, dtype=torch.long)
attention_masks_dev = encoded_data_dev['attention_mask']
attention_masks_dev = torch.tensor(attention_masks_dev, dtype=torch.long)
labels_dev = torch.tensor(labels_dev_, dtype=torch.long)









dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train[0:8000])
dataset_dev = TensorDataset(input_ids_dev, attention_masks_dev, labels_dev[0:3000])



model = BertForSequenceClassification.from_pretrained("asafaya/bert-base-arabic",
                                                      num_labels=len(label_dict))




from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 100

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_dev, 
                                   sampler=SequentialSampler(dataset_dev), 
                                   batch_size=batch_size)





from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(),
                  lr=1e-7, 
                  eps=1e-8)
                  
epochs = 3

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)





from sklearn.metrics import f1_score

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

torch.cuda.empty_cache()

model.resize_token_embeddings(len(tokenizer))

print('Loading model to GPU....')

device = torch.device("cpu")

print(' GPU:', torch.cuda.get_device_name(0))

desc = model.to(device)

'''for e in range(epochs):
    for images, labels in train_loader:   
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda() '''
            
'''del var_name
gc.collect()'''

import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.cuda.get_device_capability()
def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad(): 

            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

i=0
for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:
        print(i)

        model.zero_grad()
         
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
        
        print("training_loss: {:.3f}".format(loss.item()/len(batch)))
        
        i +=1
         
        
    torch.save(model.state_dict(), f'Project_1_tools_and_data/dataset/finetuned_BERT_epoch_.model')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')



model = BertForSequenceClassification.from_pretrained("asafaya/bert-base-arabic",
                                                      num_labels=len(label_dict))
if torch.cuda.is_available():  
    model.to(device)

model.load_state_dict(torch.load('Project_1_tools_and_data/dataset/finetuned_BERT_epoch_.model', map_location=torch.device('cpu')))

_, predictions, true_vals = evaluate(dataloader_validation)
accuracy_per_class(predictions, true_vals)




Accuracy=calculate_accuracy(predictions,true_vals)
print('Model 2 Accuracy Test: \n',Accuracy)

predicted_labels = []
for i in range(predictions.shape[0]):
    indexpred=np.argmax(predictions[i])
    predicted_labels.append(indexpred)


gold_labels = true_vals
labels = list(set(gold_labels))

accuracy = accuracy_score(gold_labels, predicted_labels) * 100 
f1 = f1_score(gold_labels, predicted_labels, labels = labels, average = None) * 100
recall = recall_score(gold_labels, predicted_labels, labels = labels, average = None) * 100
precision = precision_score(gold_labels, predicted_labels, labels = labels, average = None) * 100


print ("INDIVIDUAL PRECISION SCORE:")
for x in range (len(labels)):
    print (labels[x], ", PRECISION SCORE: %.2f" %precision[x], "%")
    
print ("\nINDIVIDUAL RECALL SCORE:")
for x in range (len(labels)):
    print (labels[x], ", RECALL SCORE: %.2f" %recall[x], "%")

print ("\nINDIVIDUAL F1 SCORE:")
for x in range (len(labels)):
    print (labels[x], ", F1 SCORE: %.2f" %f1[x], "%")



## computes overall scores (accuracy, f1, recall, precision)
accuracy = accuracy_score(gold_labels, predicted_labels) * 100
f1 = f1_score(gold_labels, predicted_labels, average = "macro") * 100
recall = recall_score(gold_labels, predicted_labels, average = "macro") * 100
precision = precision_score(gold_labels, predicted_labels, average = "macro") * 100


print ("\nOVERALL SCORES:")
## prints overall scores (accuracy, f1, recall, precision)
print ("MACRO AVERAGE PRECISION SCORE: %.2f" %precision, "%")
print ("MACRO AVERAGE RECALL SCORE: %.2f" %recall, "%")
print ("MACRO AVERAGE F1 SCORE: %.2f" %f1, "%")
print ("OVERALL ACCURACY: %.2f" %accuracy, "%\n")



labels_inverse = le1.inverse_transform(labels)










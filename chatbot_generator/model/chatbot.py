import json
import random

import chatbot_generator.model.actor as actor

import nltk
from nltk.stem.lancaster import LancasterStemmer

import numpy as np
import pandas as pd
import spacy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, InputLayer
from tensorflow.keras.optimizers import SGD

from chatbot_generator.model.translator import *

class Chatbot:
  stemmer = LancasterStemmer()

  def __init__(self, lang='EN', country='United States'):
    self.words = []
    self.classes = []
    self.documents = []
    self.ignore_words = ['?', '¿']
    self.intents = None
    self.model = None
    self.context = ""
    self.personal_information = {
      "lang": lang, 
      "country": country
    }

    self.nlp = spacy.load("en_core_web_sm")

    if(lang != 'EN'):
      generate_dataset(lang)

    self.dataset = 'chatbot_generator/dataset/covid_intents_%s.json' % lang

    with open(self.dataset) as json_data:
      self.intents = json.load(json_data)

      for intent in self.intents['intents']:
        for pattern in intent['patterns']:
          w = self.clean_tokenize_sentence(pattern)

          self.words.extend(w)
          self.documents.append( (w, intent['tag']) ) # Array< (Array<Token>, Tag) >
          if intent['tag'] not in self.classes:
            self.classes.append(intent['tag'])

    self.words = sorted(list(set(self.words))) # Array<Array<Token>>
    self.classes = sorted(list(set(self.classes))) # Array<Tags>

  def get_train_test_data(self, training):
    label_included = []
    test_x = []
    test_y = []
    train_x = []
    train_y = []

    for t in training:
      label_decoded = int("".join(str(x) for x in t[1]), 2) 

      if not label_decoded in label_included:
        train_x.append(t[0])
        train_y.append(t[1])
        label_included.append(label_decoded)
      else:
        test_x.append(t[0])
        test_y.append(t[1])
    
    return train_x, train_y, test_x, test_y

  def train_model(self):
    training = []

    output_empty = [0] * len(self.classes)

    for doc in self.documents:
      # doc is (Array<Token>, Tag)
      bag = self.bow_encode(doc[0]) # parse array token as bag of words

      output_row = list(output_empty) 
      output_row[self.classes.index(doc[1])] = 1 # parse tag as a binary

      training.append([bag, output_row]) # [ BoW, ClassBinary ]
    
    random.shuffle(training)
    training = np.array(training)

    # train_x = list(training[:,0]) # First column = Array<BoW>
    # train_y = list(training[:,1]) # Second column = Array<ClassBinary>

    train_x, train_y, test_x, test_y = self.get_train_test_data(training)

    self.model = Sequential()
    self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    self.model.add(Dropout(0.5))
    self.model.add(Dense(64, activation='relu'))
    self.model.add(Dropout(0.5))
    self.model.add(Dense(len(train_y[0]), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    self.model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1, validation_data=(train_x, train_y))

  def clean_tokenize_sentence(self, sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [self.stemmer.stem(word.lower()) for word in sentence_words if word not in self.ignore_words]
    return sentence_words

  def bow_encode(self, sentence):
    bag = [0] * len(self.words)

    for word in sentence:
      for i,w in enumerate(self.words):
        if w == word:
          bag[i] = 1

    return bag

  def predict_tag(self, question):
    clean_question = self.clean_tokenize_sentence(question)

    if (self.context != ""):
      return self.context
    else:
      inputvar =  pd.DataFrame([self.bow_encode(clean_question)], dtype=float, index=['input'])

      return self.classes[np.argmax(self.model.predict(inputvar))]

  

  def perform_response(self, prediction_tag, question):
    for intent in self.intents['intents']:
      if intent['tag'] == prediction_tag:
        self.context = random.choice(intent['context'])

        print("[%s]{%s}" % (prediction_tag, self.context), random.choice(intent['responses']))
        
        action = intent['action']

        if action:
          actor.perform_action(action, question, self.personal_information)

        return

    raise Exception("No answer available for prediction: %s" % prediction_tag)

  def ask(self, question):
    prediction_tag = self.predict_tag(question)

    self.perform_response(prediction_tag, question)

    

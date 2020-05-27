import nltk

from nltk.stem.lancaster import LancasterStemmer

import tensorflow
import tflearn
import json

stemmer = LancasterStemmer

with open('vocabs.json') as file:
    data = json.load(file)

words, labels, training, output = dump('data.joblib')
network = tflearn.input_data([None, len(training[0])])
network = tflearn.fully_connected(network, 8)
network = tflearn.fully_connected(network, 8)
network = tflearn.fully_connected(network, len(output[0]), activation='softmax')
network = tflearn.regression(network)

model = tflearn.DM(network)

model.load('model.tflearn')


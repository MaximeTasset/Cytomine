# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 16:55:56 2018

@author: Maxime
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

import numpy as np

# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
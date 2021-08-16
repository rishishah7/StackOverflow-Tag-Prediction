import tensorflow as tf

# We'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def test_model(question):

  
  with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

  train_question_seqs = tokenizer.texts_to_sequences(question)
  question_vector = tf.keras.preprocessing.sequence.pad_sequences(train_question_seqs, padding='post',maxlen=50)
  
#add model location in load_model
  model = tf.keras.models.load_model('RNN_Stack.h5')
  
  ans = model.predict(np.expand_dims(question_vector.reshape(50,1),axis = 0))
  
  for i in range(500):
    if ans[0,i]>0.95:
      ans[0,i] = 1
    else:
      ans[0,i] = 0  

  
#Selection of neurons 
  answer = []

  for i in range(500):
    if ans[0,i] == 1:
      answer.append(i)
    else:
      pass  


#add file location in mp.load()
  label_encoder = LabelEncoder()
  label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)
  ans = label_encoder.inverse_transform(answer)
  

  return ans
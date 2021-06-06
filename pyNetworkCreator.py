# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:43:24 2021

@author: xxx
"""

#this suppresses info messages in console (2 suppresses warnings, 3 suppresses errors)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2', '3'}

#suppresses deprecation warnings (not relevant, as long as tf 2.3 is used)
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf

# Helper libraries
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float32')
import tensorflow_addons as tfa

addendum="_1to1"

dataset_in = pd.read_csv("shared/MD30/250steps/1.5vel/writer2_1to1.csv", sep=";", header=None)
dataset_label = pd.read_csv("shared/MD30/250steps/1.5vel/1/writer_after1.csv", sep=";", header=None)

#"""
#this is for actual data
dataset_in=dataset_in[(dataset_in[1]!=0) & (dataset_in[1]!=13) & (dataset_in[2]!=0) & (dataset_in[2]!=13) & (dataset_in[3]!=0) & (dataset_in[3]!=13)]
dataset_in=dataset_in[[0,8]]
dataset_in.columns = range(dataset_in.shape[1])
dataset_in = dataset_in.reset_index(drop=True)

dataset_label=dataset_label[[0,8,14]]
dataset_label.columns = range(dataset_label.shape[1])
dataset_label[1].replace({0 : 1}, inplace=True)

label_array = np.empty((0, 216), float)
input_array = np.empty((0, 1512), float)

#first four steps are ignored, because unstable/inaccurate
for i in range(5, int(dataset_in.iloc[[-1]][0])):
    input_array = np.append(input_array, np.array([dataset_in[dataset_in[0]==i].transpose().iloc[1].values.tolist()]), axis=0)
    label_array = np.append(label_array, np.true_divide(np.array([dataset_label[dataset_label[0]==i].transpose().iloc[2].values.tolist()]), np.array([dataset_label[dataset_label[0]==i].transpose().iloc[1].values.tolist()])), axis=0)
#"""
"""
#this is for clean data
dataset_in=dataset_in[(dataset_in[1]!=0) & (dataset_in[1]!=13) & (dataset_in[2]!=0) & (dataset_in[2]!=13) & (dataset_in[3]!=0) & (dataset_in[3]!=13)]
dataset_in=dataset_in[[0,8]]
dataset_in.columns = range(dataset_in.shape[1])
dataset_in = dataset_in.reset_index(drop=True)

dataset_label=dataset_label[[0,4]]
dataset_label.columns = range(dataset_label.shape[1])

label_array = np.empty((0, 216), float)
input_array = np.empty((0, 1512), float)

#first four steps are ignored, because unstable/inaccurate
for i in range(5, int(dataset_in.iloc[[-1]][0])):
    input_array = np.append(input_array, np.array([dataset_in[dataset_in[0]==i].transpose().iloc[1].values.tolist()]), axis=0)
    label_array = np.append(label_array, np.array([dataset_label[dataset_label[0]==i].transpose().iloc[1].values.tolist()]), axis=0)
"""
"""
indices = tf.range(start=0, limit=tf.shape(input_array)[0], dtype=tf.int32)
shuffled_indices = tf.random.shuffle(indices)

shuffled_x = tf.gather(input_array, shuffled_indices)
shuffled_y = tf.gather(label_array, shuffled_indices)
"""

model = tf.keras.models.Sequential([
  tf.keras.layers.InputLayer(input_shape=input_array[0].shape[0]),
  tf.keras.layers.Dense(units=input_array[0].shape[0], activation='relu'),
  tf.keras.layers.Dense(units=label_array[0].shape[0])
])

loss_fn = tf.keras.losses.MeanSquaredError(reduction="auto")
lr = 0.0000005

opt=tf.keras.optimizers.Adam(
    learning_rate=lr,
    name='adam_opt'
)

model.compile(optimizer=opt,
              loss=loss_fn,
              metrics=[tf.metrics.MeanAbsoluteError()])

model.fit(input_array, label_array, batch_size=1, epochs=100)

tf.keras.models.save_model(model, 'shared/model')



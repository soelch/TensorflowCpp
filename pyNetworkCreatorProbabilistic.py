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
import tensorflow_probability as tfp

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float32')
import tensorflow_addons as tfa
import seaborn as sns

sns.reset_defaults()
#sns.set_style('whitegrid')
#sns.set_context('talk')
sns.set_context(context='talk',font_scale=0.7)

tfd = tfp.distributions

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return lambda t: tfd.Independent(tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=1), 
                                     reinterpreted_batch_ndims=1)

def prior_trainable(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(n+1, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t[...,:n], scale=t[...,n:]*0.001+1e-10),
          reinterpreted_batch_ndims=1)),
  ])


def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
            tfp.layers.VariableLayer(tfp.layers.IndependentNormal.params_size(n), dtype=dtype),
            tfp.layers.IndependentNormal(n)
        ])

def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  c = np.log(np.expm1(1e-5))
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(2 * n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t[..., :n],
                     scale=1e-10 + tf.nn.softplus(c + t[..., n:])),
          reinterpreted_batch_ndims=1)),
  ])

#all of these need to contain the same amount of timesteps
datasets_in = ["shared/MD30/250steps/1.5vel/writer2_1to1.csv"]
datasets_label = ["shared/MD30/250steps/1.5vel/1/writer_after1.csv", "shared/MD30/250steps/1.5vel/2/writer_after1.csv", "shared/MD30/250steps/1.5vel/3/writer_after1.csv"]

dflist_in = list()
dflist_label = list()

for dataset_in in datasets_in:
    dflist_in.append(pd.read_csv(dataset_in, sep=";", header=None))
    
for dataset_label in datasets_label:
    dflist_label.append(pd.read_csv(dataset_label, sep=";", header=None))


#"""
#this is for actual data
for i in range(len(dflist_in)):
    dflist_in[i]=dflist_in[i][(dflist_in[i][1]!=0) & (dflist_in[i][1]!=13) & (dflist_in[i][2]!=0) & (dflist_in[i][2]!=13) & (dflist_in[i][3]!=0) & (dflist_in[i][3]!=13)]
    dflist_in[i]=dflist_in[i][[0,8]]
    dflist_in[i].columns = range(dflist_in[i].shape[1])
    dflist_in[i] = dflist_in[i].reset_index(drop=True)

for i in range(len(dflist_label)):
    dflist_label[i]=dflist_label[i][[0,8,14]]
    dflist_label[i].columns = range(dflist_label[i].shape[1])
    #this is needed for some weird issue, where some masses are set to zero (might have been a problem with using the wrong savepoint for md)
    dflist_label[i][1].replace({0 : 1}, inplace=True)



label_array = np.empty((0, 216), float)
input_array = np.empty((0, 1512), float)

n_datasets= len(dflist_label)

for i in range(1, int(dflist_in[0].iloc[[-1]][0])+1):
    for j in range(n_datasets):
        input_array = np.append(input_array, np.array([dflist_in[0][dflist_in[0][0]==i].transpose().iloc[1].values.tolist()]), axis=0)
        label_array = np.append(label_array, np.true_divide(np.array([dflist_label[j][dflist_label[j][0]==i].transpose().iloc[2].values.tolist()]), np.array([dflist_label[j][dflist_label[j][0]==i].transpose().iloc[1].values.tolist()])), axis=0)


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
  tf.keras.layers.InputLayer(input_shape=(np.shape(input_array)[1],), name="input"),
  tfp.layers.DenseVariational(np.shape(label_array)[1]+1, posterior_mean_field, prior_trainable, kl_weight=0.1/input_array.shape[0],#/batch_size,
                              kl_use_exact=False),
  tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :np.shape(label_array)[1]], scale=1e-6 + t[...,np.shape(label_array)[1]:]*0.0001)),
])

loss_fn = tf.keras.losses.MeanSquaredError(reduction="auto")
lr = 0.005

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    lr,
    decay_steps=200,
    decay_rate=0.99,
    staircase=False)

opt=tf.keras.optimizers.Adam(
    learning_rate=lr_schedule,
    name='adam_opt'
)

negloglik = lambda y, y_pred: -y_pred.log_prob(y)

model.compile(optimizer=opt,
              loss=negloglik,
              metrics=[tf.metrics.MeanAbsoluteError()])

model.fit(input_array, label_array, batch_size=1, epochs=250)

tf.keras.models.save_model(model, 'shared/model_prob_batch'+str(n_datasets))



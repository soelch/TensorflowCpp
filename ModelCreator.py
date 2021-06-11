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
tf.keras.backend.set_floatx('float32')
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

#loads all csv files specified in the lists and arranges them by timestep, so that n_scenarios*n_datasets can be chosen as batch size to hopefully average the error in each time step
def get_real_trainig_data(datasets_in, datasets_label):
    dflist_in = []
    dflist_label = []
    
    if(len(datasets_in)!=len(datasets_label)):
        print("size of inputs does not match size of labels")
        exit(1)
    tmp=len(datasets_label[0])
    for i in range(1,len(datasets_label)):
        if(len(datasets_label[i])!=tmp):
            print("not all scenarios are the same length")
            exit(2)
    
    for scenario_in in datasets_in:
        for dataset_in in scenario_in:
            dflist_in.append(pd.read_csv(dataset_in, sep=";", header=None))
        
    for scenario_label in datasets_label:
        for dataset_label in scenario_label:
            dflist_label.append(pd.read_csv(dataset_label, sep=";", header=None))
        
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
    
    n_scenarios = len(datasets_label)
    n_datasets = len(datasets_label[0])
    
    for i in range(1, int(dflist_in[0].iloc[[-1]][0])+1):
        for j in range(n_scenarios):
            tmp=j*n_scenarios
            for k in range(n_datasets):
                input_array = np.append(input_array, np.array([dflist_in[j][dflist_in[j][0]==i].transpose().iloc[1].values.tolist()]), axis=0)
                label_array = np.append(label_array, np.true_divide(np.array([dflist_label[tmp+k][dflist_label[tmp+k][0]==i].transpose().iloc[2].values.tolist()]), np.array([dflist_label[tmp+k][dflist_label[tmp+k][0]==i].transpose().iloc[1].values.tolist()])), axis=0)

    return input_array, label_array, n_scenarios, n_datasets;

def get_clean_trainig_data():
    dataset_in=pd.read_csv("MD30/clean/writer2_clean.csv", sep=";", header=None)
    dataset_label=pd.read_csv("MD30/clean/writer1_clean.csv", sep=";", header=None)
    
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

def compile_prob_model(size_in, size_out, n_datasets):
    model = tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(size_in,), name="input"),
      tfp.layers.DenseVariational(size_out+1, posterior_mean_field, prior_trainable, kl_weight=0.1/(250*6),#/batch_size <-- should this be batch size or total dataset/epoch size?
                                  kl_use_exact=False),
      tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :size_out], scale=1e-6 + t[...,size_out:]*0.000001)),
    ])
    
    lr = 0.0005
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        lr,
        decay_steps=400,
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
    
    return model

def compile_prob_model_modified(size_in, size_out, n_datasets):
    model = tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(size_in,), name="input"),
      tfp.layers.DenseVariational(size_out+1, posterior_mean_field, prior_trainable, kl_weight=1/(250*12),#/batch_size <-- should this be batch size or total dataset/epoch size?
                                  kl_use_exact=False),
      tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :size_out], scale=1e-7 + tf.nn.softplus(t[...,size_out:]*0.05))),
    ])
    
    lr = 0.0005
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        lr,
        decay_steps=500,
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
    
    return model

def compile_std_model(size_in, size_out):
    model = tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=size_in),
      tf.keras.layers.Dense(units=size_in, activation='relu'),  #<-- check, if size_out is sufficient
      tf.keras.layers.Dense(units=size_out)
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
    
    return model
    
def prob_run(datasets_in, datasets_label, datatype, ep):
    input_array, label_array, n_scenarios, n_datasets = get_real_trainig_data(datasets_in, datasets_label)

    model = compile_prob_model(np.shape(input_array)[1], np.shape(label_array)[1], n_datasets)
    
    model.fit(input_array, label_array, batch_size=n_datasets*n_scenarios, epochs=ep, shuffle=True)
    
    tf.keras.models.save_model(model, 'model_prob_'+datatype+'_s'+str(n_scenarios)+'_b'+str(n_datasets))

def prob_run_modified(datasets_in, datasets_label, datatype, ep):
    input_array, label_array, n_scenarios, n_datasets = get_real_trainig_data(datasets_in, datasets_label)

    model = compile_prob_model_modified(np.shape(input_array)[1], np.shape(label_array)[1], n_datasets)
    
    model.fit(input_array, label_array, batch_size=n_datasets*n_scenarios, epochs=ep, shuffle=True)
    
    tf.keras.models.save_model(model, 'model_prob_'+datatype+'_s'+str(n_scenarios)+'_b'+str(n_datasets))

def std_run(datasets_in, datasets_label, datatype, ep):
    input_array, label_array, n_scenarios, n_datasets = get_real_trainig_data(datasets_in, datasets_label)

    model = compile_std_model(np.shape(input_array)[1], np.shape(label_array)[1])
    
    model.fit(input_array, label_array, batch_size=n_datasets*n_scenarios, epochs=ep)
    
    tf.keras.models.save_model(model, 'model_std_'+datatype+'_s'+str(n_scenarios)+'_b'+str(n_datasets))

##########################################

#all of these need to contain the same amount of timesteps
#also each set of datasets(i.e. sets of same sim settings) should contain same amount of datasets
#otherwise, the batch size will not function as intended

datatype="30"
gauss=0

if(gauss==0):
    gauss="before"
else:
    gauss="after"+str(gauss)

if(datatype=="30"):
    datasets_in = [["../shared/MD30/250steps/1.5vel/writer2.csv"], 
                   ["../shared/MD30/250steps/1.0vel/writer2.csv"]
                   ]

    datasets_label = [["../shared/MD30/250steps/1.5vel/1/writer_"+gauss+".csv", 
                       "../shared/MD30/250steps/1.5vel/2/writer_"+gauss+".csv", 
                       "../shared/MD30/250steps/1.5vel/3/writer_"+gauss+".csv",
                       "../shared/MD30/250steps/1.5vel/4/writer_"+gauss+".csv", 
                       "../shared/MD30/250steps/1.5vel/5/writer_"+gauss+".csv", 
                       "../shared/MD30/250steps/1.5vel/6/writer_"+gauss+".csv"],
                      
                      ["../shared/MD30/250steps/1.0vel/1/writer_"+gauss+".csv", 
                       "../shared/MD30/250steps/1.0vel/2/writer_"+gauss+".csv", 
                       "../shared/MD30/250steps/1.0vel/3/writer_"+gauss+".csv",
                       "../shared/MD30/250steps/1.0vel/4/writer_"+gauss+".csv", 
                       "../shared/MD30/250steps/1.0vel/5/writer_"+gauss+".csv", 
                       "../shared/MD30/250steps/1.0vel/6/writer_"+gauss+".csv"]
                      ]
if(datatype=="60"):
    datasets_in = [["../shared/MD60/500steps/1.5vel/writer2.csv"], 
                   ["../shared/MD60/500steps/1.0vel/writer2.csv"]
                   ]

    datasets_label = [["../shared/MD60/500steps/1.5vel/1/writer60_"+gauss+".csv", 
                       "../shared/MD60/500steps/1.5vel/2/writer60_"+gauss+".csv", 
                       "../shared/MD60/500steps/1.5vel/3/writer60_"+gauss+".csv",
                      # "../shared/MD60/500steps/1.5vel/4/writer60_"+gauss+".csv", 
                      # "../shared/MD60/500steps/1.5vel/5/writer60_"+gauss+".csv", 
                      # "../shared/MD60/500steps/1.5vel/6/writer60_"+gauss+".csv"
                      ],
                      
                      ["../shared/MD60/500steps/1.0vel/1/writer60_"+gauss+".csv", 
                       "../shared/MD60/500steps/1.0vel/2/writer60_"+gauss+".csv", 
                       "../shared/MD60/500steps/1.0vel/3/writer60_"+gauss+".csv",
                       #"../shared/MD60/500steps/1.0vel/4/writer60_"+gauss+".csv"
                       ]
                      ]
    
prob_run_modified(datasets_in, datasets_label, datatype, 1500)




"""
import random

indices = tf.range(start=0, limit=tf.shape(input_array)[0], dtype=tf.int32)
shuffled_indices = tf.random.shuffle(indices)

shuffled_x = tf.gather(input_array, shuffled_indices)
shuffled_y = tf.gather(label_array, shuffled_indices)
"""

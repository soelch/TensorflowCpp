# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:43:24 2021

@author: xxx
"""

#this suppresses info messages in console (2 suppresses warnings, 3 suppresses errors)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2', '3'}

import time

#suppresses deprecation warnings (not relevant, as long as tf 2.3 is used)
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import pandas as pd
tf.keras.backend.set_floatx('float32')
import seaborn as sns
from matplotlib import pyplot as plt
import math


sns.reset_defaults()
#sns.set_style('whitegrid')
#sns.set_context('talk')
sns.set_context(context='talk',font_scale=0.7)

tfd = tfp.distributions

class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        # use this value as reference to calculate cummulative time taken
        self.current = time.process_time()
    def on_epoch_end(self,epoch,logs = {}):
        self.times.append(time.process_time() - self.current)
        self.current = time.process_time()
    def on_train_end(self,logs = {}):
        print("Average Time: "+str(np.sum(self.times)/len(self.times)))
        print("Total Time: "+str(np.sum(self.times)))
        print(len(self.times))
        
class LearningRateLoggingCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch,logs={}):
        print("lr: ")
        print(self.model.optimizer._decayed_lr(tf.float32).numpy())
        if(self.model.optimizer._decayed_lr(tf.float32).numpy()<2e-6):
            self.model.optimizer.lr.decay_rate=1
            self.model.optimizer.lr.initial_learning_rate=2e-6
       

def plot(hist):
    plt.plot(hist)
    plt.xticks(np.arange(len(hist)), np.arange(1, len(hist)+1))
    plt.yscale("log")
    plt.show()

#this gets two losts that are the same dimension as the network in/output
#the xlsx files are formatted to only include the velocities in the correct order
def get_comparison_data():
    d_in=pd.read_excel("../shared/MD30/200_1.5_in.xlsx", header=None)
    d_in=np.array(d_in.transpose().values.tolist())#[0]
    d_out=pd.read_excel("../shared/MD30/200_1.5_out.xlsx", header=None)
    d_out=np.array(d_out.transpose().values.tolist()[0])
    return d_in, d_out;

def run_comparison_prob(model,b,kl):
    d_in, d_out= get_comparison_data()
    res=model.predict(d_in)[0]
    plt.plot(d_out)
    plt.plot(res)
    plt.show()
    
    plt.plot(model(d_in).stddev()[0])
    plt.show()
    
    res=np.mean(np.array_split(model(d_in).stddev()[0],6), axis=1)
    plt.plot(res)
    plt.show()
    
    plt.plot(d_out)
    res=model(d_in).mean()[0]
    plt.plot(res,'go')
    plt.show()
    
    plt.plot(d_out)
    plt.plot(np.arange(17,216,36), np.mean(np.array_split(res,6), axis=1),'go')
    plt.show()
    
    for i in range(99):
        res+=model(d_in).mean()[0]
    res=res*0.01
    plt.plot(d_out)
    res=np.mean(np.array_split(res,6), axis=1)
    plt.plot(np.arange(17,216,36), res,'go')
    plt.show()

def run_comparison_std(model,b,kl):
    d_in, d_out= get_comparison_data()
    res=model.predict(d_in)[0]

    # res=np.expand_dims(res,axis=0)
    # res=np.mean(res, axis=0)
    plt.plot(d_out)
    plt.plot(res)
    plt.show()
    
    res=np.mean(np.array_split(res,6), axis=1)
    plt.plot(d_out)
    plt.plot(np.arange(17,216,36),res,'go')
    plt.show()
    

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return lambda t: tfd.Independent(tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=.1), 
                                     reinterpreted_batch_ndims=1)

def prior_trainable(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  c = np.log(np.expm1(1))
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(2*n, dtype=dtype, 
                               #initializer=tfp.layers.BlockwiseInitializer(['zeros',
                               #tf.keras.initializers.Constant(np.log(np.expm1(1.)))], sizes=[n, n])
                               ),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t[...,:n], scale=tf.nn.softplus(c+t[..., n:])),
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
  c = np.log(np.expm1(1))#e-7
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(2*n, dtype=dtype, 
                               #initializer=tfp.layers.BlockwiseInitializer(['zeros',
                               #tf.keras.initializers.Constant(np.log(np.expm1(1.)))], sizes=[n, n])
                               ),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t[..., :n]+0.2, scale=0.00001*tf.nn.softplus(c + t[..., n:])),
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
            tmp=j*n_datasets
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

def act(x):
    return x*x*x+100*x

act_vectorize=np.vectorize(act)
act_convert= lambda x: act_vectorize(x).astype(np.float32)

def act_final(arg):
    return tf.convert_to_tensor(arg, dtype=tf.float32)

def compile_prob_model_modified(size_in, size_out, n_datasets, kl_mod):
    model = tf.keras.models.Sequential([
      tf.keras.layers.InputLayer(input_shape=(size_in,), name="input"),
      tfp.layers.DenseVariational(size_out*2, posterior_mean_field, prior_trainable, kl_weight=kl_mod, #/batch_size <-- should this be batch size or total dataset/epoch size?
                                   kl_use_exact=True, activation="relu"),#tf.keras.layers.LeakyReLU(alpha=0.01)
      #tf.keras.layers.Dense(2*size_out, activation=act_final, kernel_initializer=tf.keras.initializers.RandomUniform(minval=1/1500, maxval=2/1500),),
      tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :size_out], scale=tf.nn.softplus(t[...,size_out:]))),
    ])
    
    lr = 0.000005
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        lr,
        decay_steps=100,
        decay_rate=0.95,
        staircase=False)
    
    opt=tf.keras.optimizers.Adam(
        learning_rate=lr,
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
      tf.keras.layers.Dense(units=size_out/4, activation="relu"),#act_final
      tf.keras.layers.Dense(units=size_out/4, activation="relu"),#act_final
      tf.keras.layers.Dense(units=size_out)
    ])

    
    loss_fn = tf.keras.losses.MeanSquaredError(reduction="auto")
    lr = 0.0000005
    
    lrs= tf.keras.optimizers.schedules.ExponentialDecay(
        lr,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=False)
    
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

def prob_run_modified(datasets_in, datasets_label, datatype, ep, batch_sizes, kl_mod):
    if(len(ep)!=len(batch_sizes)):
        print("ep and batch_size do not match")
        return
    
    input_array, label_array, n_scenarios, n_datasets = get_real_trainig_data(datasets_in, datasets_label)

    model = compile_prob_model_modified(np.shape(input_array)[1], np.shape(label_array)[1], n_datasets, kl_mod)
    
    timetaken = timecallback()
    lrcall=LearningRateLoggingCallback()
    
    loss=[]
    mae=[]

    for i in range(len(ep)):
        temp=model.fit(input_array, label_array, batch_size=batch_sizes[i], epochs=ep[i], shuffle=True,callbacks = [lrcall,timetaken])
        run_comparison_prob(model,batch_sizes[i],kl_mod)
        
        loss.extend(temp.history["loss"])
        mae.extend(temp.history["mean_absolute_error"])
    #plot(loss)
    
    tf.keras.models.save_model(model, 'model_prob_'+datatype+'_s'+str(n_scenarios)+'_b'+str(n_datasets)+"_final_1")

def std_run(datasets_in, datasets_label, datatype, ep, batch_sizes):
    input_array, label_array, n_scenarios, n_datasets = get_real_trainig_data(datasets_in, datasets_label)
    
    timetaken = timecallback()
    lrcall=LearningRateLoggingCallback()

    model = compile_std_model(np.shape(input_array)[1], np.shape(label_array)[1])
    loss=[]
    mae=[]
    kl_mod=0
    for i in range(len(ep)):
        temp=model.fit(input_array, label_array, batch_size=batch_sizes[i], epochs=ep[i], shuffle=True,callbacks = [lrcall,timetaken])
        run_comparison_std(model,batch_sizes[i],kl_mod)
        loss.extend(temp.history["loss"])
        mae.extend(temp.history["mean_absolute_error"])
    #plot(loss)
    
    tf.keras.models.save_model(model, 'model_std_'+datatype+'_s'+str(n_scenarios)+'_b'+str(n_datasets)+"_sp")
    
def compare(tstep, vel, mtype, mdtype, ndata, ndtype, additional):
    a=np.expand_dims(np.genfromtxt("../shared/MD30/analytical/"+str(vel)+"/"+str(tstep)+"_in.csv"),axis=0)
    comp=np.genfromtxt("../shared/MD30/analytical/"+str(vel)+"/"+str(tstep)+"_comp.csv")
    
    model = tf.keras.models.load_model("model_"+mtype+"_"+mdtype+"_s"+str(ndtype)+"_b"+str(ndata)+additional)

    res=model.predict(a)
    plt.plot(res[0])
    plt.show()
    plt.plot(np.arange(11.25,23.75+1,2.5),np.mean(np.array_split(res[0],6), axis=1),"go")
    plt.plot(np.arange(0,50.1,0.1),comp)
    plt.show()
    print(np.std(np.array_split(res[0],6), axis=1))

def writePred(tstep, vel, mtype, mdtype, ndata, ndtype, additional):
    if(mtype=="prob"):
        negloglik = lambda y, y_pred: -y_pred.log_prob(y)
        model = tf.keras.models.load_model("model_"+mtype+"_"+mdtype+"_s"+str(ndtype)+"_b"+str(ndata)+additional, custom_objects={'<lambda>' : negloglik })
    else:
        model = tf.keras.models.load_model("model_"+mtype+"_"+mdtype+"_s"+str(ndtype)+"_b"+str(ndata)+additional)

    for i in range(50,251,50):
        path="../shared/MD30/predictions/"+mtype+"/final2/"+str(vel)+additional+"/"#"/"+str(ndtype)+"_"+str(ndata)+
        if not os.path.exists(path):
            os.mkdir(path)
        path=path+str(i)
        comp=np.genfromtxt("../shared/MD30/analytical/"+str(vel)+"/"+str(i)+"_comp.csv", delimiter=",")
        comp2=[]
        comp2.append((comp[112][1]+comp[113][1])/2)
        comp2.append((comp[137][1]+comp[138][1])/2)
        comp2.append((comp[162][1]+comp[163][1])/2)
        comp2.append((comp[187][1]+comp[188][1])/2)
        comp2.append((comp[212][1]+comp[213][1])/2)
        comp2.append((comp[237][1]+comp[238][1])/2)
        comp=np.array(comp2)
        res=model.predict(np.expand_dims(np.genfromtxt("../shared/MD30/analytical/"+str(vel)+"/"+str(i)+"_in.csv"),axis=0))[0]
        np.savetxt(path+"_raw.csv", np.transpose(np.vstack((np.arange(1,217,1),res))),header="x,y",comments='',delimiter=",", fmt='%f')
        res=model.predict(np.expand_dims(np.genfromtxt("../shared/MD30/analytical/"+str(vel)+"/"+str(i)+"_in.csv"),axis=0))[0]
        np.savetxt(path+"_raw2.csv", np.transpose(np.vstack((np.arange(1,217,1),res))),header="x,y",comments='',delimiter=",", fmt='%f')
        
        if(mtype=="prob"):
            mean=model(np.expand_dims(np.genfromtxt("../shared/MD30/analytical/"+str(vel)+"/"+str(i)+"_in.csv"),axis=0)).mean()[0]
            np.savetxt(path+"_rawmean.csv", np.transpose(np.vstack((np.arange(1,217,1),mean))),header="x,y",comments='',delimiter=",", fmt='%f')
            np.savetxt(path+"_mean.csv", np.transpose(np.vstack((np.arange(11.25,23.75+1,2.5),np.mean(np.array_split(mean,6), axis=1)))),header="x,y",comments='',delimiter=",", fmt='%f')
            stddev=model(np.expand_dims(np.genfromtxt("../shared/MD30/analytical/"+str(vel)+"/"+str(i)+"_in.csv"),axis=0)).stddev()[0]
            np.savetxt(path+"_absErrMean.csv", np.expand_dims(np.mean(np.absolute(np.mean(np.array_split(mean,6), axis=1)-comp)),axis=0),header="x,y",comments='',delimiter=",", fmt='%f')
            print(str(i)+": "+str(np.mean(np.absolute(np.mean(np.array_split(mean,6), axis=1)-comp))))
            np.savetxt(path+"meanstddev.csv", np.expand_dims(np.mean(stddev),axis=0),delimiter=",", fmt='%f')
            for j in range(99):
                mean+=model(np.expand_dims(np.genfromtxt("../shared/MD30/analytical/"+str(vel)+"/"+str(i)+"_in.csv"),axis=0)).mean()[0]
            mean=mean*0.01
            np.savetxt(path+"_meanmean.csv", np.transpose(np.vstack((np.arange(11.25,23.75+1,2.5),np.mean(np.array_split(mean,6), axis=1)))),header="x,y",comments='',delimiter=",", fmt='%f')
            np.savetxt(path+"_absErrMeanMean.csv", np.expand_dims(np.mean(np.absolute(np.mean(np.array_split(mean,6), axis=1)-comp)),axis=0),header="x,y",comments='',delimiter=",", fmt='%f')
            
        else:
            np.savetxt(path+"_mean.csv", np.transpose(np.vstack((np.arange(11.25,23.75+1,2.5),np.mean(np.array_split(res,6), axis=1)))),header="x,y",comments='',delimiter=",", fmt='%f')
            np.savetxt(path+"_absErr.csv", np.transpose(np.vstack((np.arange(11.25,23.75+1,2.5),np.mean(np.array_split(res,6), axis=1)-comp))),header="x,y",comments='',delimiter=",", fmt='%f')
            np.savetxt(path+"_absErrMean.csv", np.expand_dims(np.mean(np.absolute(np.mean(np.array_split(res,6), axis=1)-comp)),axis=0),header="x,y",comments='',delimiter=",", fmt='%f')
            print(str(i)+": "+str(np.mean(np.absolute(np.mean(np.array_split(res,6), axis=1)-comp))))
            np.savetxt(path+"_stddev.csv", np.std(np.array_split(res,6), axis=1),delimiter=",", fmt='%f')
            np.savetxt(path+"meanstddev.csv", np.expand_dims(np.mean(np.std(np.array_split(res,6), axis=1)),axis=0),delimiter=",", fmt='%f')
            

    
    for i in range(300,1001,50):
        path="../shared/MD30/predictions/"+mtype+"/final2/"+str(vel)+additional+"/"#"/"+str(ndtype)+"_"+str(ndata)+
        if not os.path.exists(path):
            os.mkdir(path)
        path=path+str(i)
        comp=np.genfromtxt("../shared/MD30/analytical/"+str(vel)+"/"+str(i)+"_comp.csv", delimiter=",")
        comp2=[]
        comp2.append((comp[112][1]+comp[113][1])/2)
        comp2.append((comp[137][1]+comp[138][1])/2)
        comp2.append((comp[162][1]+comp[163][1])/2)
        comp2.append((comp[187][1]+comp[188][1])/2)
        comp2.append((comp[212][1]+comp[213][1])/2)
        comp2.append((comp[237][1]+comp[238][1])/2)
        comp=np.array(comp2)
        res=model.predict(np.expand_dims(np.genfromtxt("../shared/MD30/analytical/"+str(vel)+"/"+str(i)+"_in.csv"),axis=0))[0]
        np.savetxt(path+"_raw.csv", np.transpose(np.vstack((np.arange(1,217,1),res))),header="x,y",comments='',delimiter=",", fmt='%f')
        if(mtype=="prob"):
            mean=model(np.expand_dims(np.genfromtxt("../shared/MD30/analytical/"+str(vel)+"/"+str(i)+"_in.csv"),axis=0)).mean()[0]
            np.savetxt(path+"_rawmean.csv", np.transpose(np.vstack((np.arange(1,217,1),mean))),header="x,y",comments='',delimiter=",", fmt='%f')
            np.savetxt(path+"_mean.csv", np.transpose(np.vstack((np.arange(11.25,23.75+1,2.5),np.mean(np.array_split(mean,6), axis=1)))),header="x,y",comments='',delimiter=",", fmt='%f')
            stddev=model(np.expand_dims(np.genfromtxt("../shared/MD30/analytical/"+str(vel)+"/"+str(i)+"_in.csv"),axis=0)).stddev()[0]
            np.savetxt(path+"_absErrMean.csv", np.expand_dims(np.mean(np.absolute(np.mean(np.array_split(mean,6), axis=1)-comp)),axis=0),header="x,y",comments='',delimiter=",", fmt='%f')
            print(str(i)+": "+str(np.mean(np.absolute(np.mean(np.array_split(mean,6), axis=1)-comp))))
            
            print(np.mean(stddev))
            np.savetxt(path+"meanstddev.csv", np.expand_dims(np.mean(stddev),axis=0),delimiter=",", fmt='%f')
            for j in range(99):
                mean+=model(np.expand_dims(np.genfromtxt("../shared/MD30/analytical/"+str(vel)+"/"+str(i)+"_in.csv"),axis=0)).mean()[0]
            mean=mean*0.01
            np.savetxt(path+"_meanmean.csv", np.transpose(np.vstack((np.arange(11.25,23.75+1,2.5),np.mean(np.array_split(mean,6), axis=1)))),header="x,y",comments='',delimiter=",", fmt='%f')
            np.savetxt(path+"_absErrMeanMean.csv", np.expand_dims(np.mean(np.absolute(np.mean(np.array_split(mean,6), axis=1)-comp)),axis=0),header="x,y",comments='',delimiter=",", fmt='%f')
            
        else:
            np.savetxt(path+"_mean.csv", np.transpose(np.vstack((np.arange(11.25,23.75+1,2.5),np.mean(np.array_split(res,6), axis=1)))),header="x,y",comments='',delimiter=",", fmt='%f')
            np.savetxt(path+"_absErr.csv", np.transpose(np.vstack((np.arange(11.25,23.75+1,2.5),np.mean(np.array_split(res,6), axis=1)-comp))),header="x,y",comments='',delimiter=",", fmt='%f')
            np.savetxt(path+"_absErrMean.csv", np.expand_dims(np.mean(np.absolute(np.mean(np.array_split(res,6), axis=1)-comp)),axis=0),header="x,y",comments='',delimiter=",", fmt='%f')
            print(str(i)+": "+str(np.mean(np.absolute(np.mean(np.array_split(res,6), axis=1)-comp))))
            np.savetxt(path+"_stddev.csv", np.std(np.array_split(res,6), axis=1),delimiter=",", fmt='%f')
            np.savetxt(path+"meanstddev.csv", np.expand_dims(np.mean(np.std(np.array_split(res,6), axis=1)),axis=0),delimiter=",", fmt='%f')
    
    for i in range(1500,5001,500):
        path="../shared/MD30/predictions/"+mtype+"/final2/"+str(vel)+additional+"/"#"/"+str(ndtype)+"_"+str(ndata)+
        if not os.path.exists(path):
            os.mkdir(path)
        path=path+str(i)
        comp=np.genfromtxt("../shared/MD30/analytical/"+str(vel)+"/"+str(i)+"_comp.csv", delimiter=",")
        comp2=[]
        comp2.append((comp[112][1]+comp[113][1])/2)
        comp2.append((comp[137][1]+comp[138][1])/2)
        comp2.append((comp[162][1]+comp[163][1])/2)
        comp2.append((comp[187][1]+comp[188][1])/2)
        comp2.append((comp[212][1]+comp[213][1])/2)
        comp2.append((comp[237][1]+comp[238][1])/2)
        comp=np.array(comp2)
        res=model.predict(np.expand_dims(np.genfromtxt("../shared/MD30/analytical/"+str(vel)+"/"+str(i)+"_in.csv"),axis=0))[0]
        np.savetxt(path+"_raw.csv", np.transpose(np.vstack((np.arange(1,217,1),res))),header="x,y",comments='',delimiter=",", fmt='%f')
        if(mtype=="prob"):
            mean=model(np.expand_dims(np.genfromtxt("../shared/MD30/analytical/"+str(vel)+"/"+str(i)+"_in.csv"),axis=0)).mean()[0]
            np.savetxt(path+"_rawmean.csv", np.transpose(np.vstack((np.arange(1,217,1),mean))),header="x,y",comments='',delimiter=",", fmt='%f')
            np.savetxt(path+"_mean.csv", np.transpose(np.vstack((np.arange(11.25,23.75+1,2.5),np.mean(np.array_split(mean,6), axis=1)))),header="x,y",comments='',delimiter=",", fmt='%f')
            stddev=model(np.expand_dims(np.genfromtxt("../shared/MD30/analytical/"+str(vel)+"/"+str(i)+"_in.csv"),axis=0)).stddev()[0]
            np.savetxt(path+"_absErrMean.csv", np.expand_dims(np.mean(np.absolute(np.mean(np.array_split(mean,6), axis=1)-comp)),axis=0),header="x,y",comments='',delimiter=",", fmt='%f')
            print(str(i)+": "+str(np.mean(np.absolute(np.mean(np.array_split(mean,6), axis=1)-comp))))
            
            print(np.mean(stddev))
            np.savetxt(path+"meanstddev.csv", np.expand_dims(np.mean(stddev),axis=0),delimiter=",", fmt='%f')
            for j in range(99):
                mean+=model(np.expand_dims(np.genfromtxt("../shared/MD30/analytical/"+str(vel)+"/"+str(i)+"_in.csv"),axis=0)).mean()[0]
            mean=mean*0.01
            np.savetxt(path+"_meanmean.csv", np.transpose(np.vstack((np.arange(11.25,23.75+1,2.5),np.mean(np.array_split(mean,6), axis=1)))),header="x,y",comments='',delimiter=",", fmt='%f')
            np.savetxt(path+"_absErrMeanMean.csv", np.expand_dims(np.mean(np.absolute(np.mean(np.array_split(mean,6), axis=1)-comp)),axis=0),header="x,y",comments='',delimiter=",", fmt='%f')
            
        else:
            np.savetxt(path+"_mean.csv", np.transpose(np.vstack((np.arange(11.25,23.75+1,2.5),np.mean(np.array_split(res,6), axis=1)))),header="x,y",comments='',delimiter=",", fmt='%f')
            np.savetxt(path+"_absErr.csv", np.transpose(np.vstack((np.arange(11.25,23.75+1,2.5),np.mean(np.array_split(res,6), axis=1)-comp))),header="x,y",comments='',delimiter=",", fmt='%f')
            np.savetxt(path+"_absErrMean.csv", np.expand_dims(np.mean(np.absolute(np.mean(np.array_split(res,6), axis=1)-comp)),axis=0),header="x,y",comments='',delimiter=",", fmt='%f')
            print(str(i)+": "+str(np.mean(np.absolute(np.mean(np.array_split(res,6), axis=1)-comp))))
            np.savetxt(path+"_stddev.csv", np.std(np.array_split(res,6), axis=1),delimiter=",", fmt='%f')
            np.savetxt(path+"meanstddev.csv", np.expand_dims(np.mean(np.std(np.array_split(res,6), axis=1)),axis=0),delimiter=",", fmt='%f')

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
                        "../shared/MD30/250steps/1.5vel/6/writer_"+gauss+".csv"
                        ],
                      
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




ep=[2500]
batch_sizes=[100]
kl_mod=1/(3000/1)# /10
l=1
a=1
n=1512*a


tstep=200
vel=1.5
mtype="prob"
mdtype="30"
ndtype=2
ndata=6


#std_run(datasets_in, datasets_label, datatype, ep, batch_sizes)




#prob_run_modified(datasets_in, datasets_label, datatype, ep, batch_sizes, kl_mod)

additional="_final_good"
writePred(tstep, vel, mtype, mdtype, ndata, ndtype, additional)


#compare(tstep, vel, mtype, mdtype, ndata, ndtype, additional)


# =============================================================================
# input_array, label_array, n_scenarios, n_datasets = get_real_trainig_data(datasets_in, datasets_label)
# d_in, d_out= get_comparison_data()
#    
# plt.plot(d_out)
# res=np.mean(np.array_split(np.mean(label_array[1194:1200], axis=0),6), axis=1)
# plt.plot(np.arange(17,216,36),res,'go')
# plt.show()
# =============================================================================






# ep=[1000,1000,1000,10000]
# batch_sizes=[250,250,250,250]
# std_run(datasets_in, datasets_label, datatype, ep, batch_sizes)


"""
import random

indices = tf.range(start=0, limit=tf.shape(input_array)[0], dtype=tf.int32)
shuffled_indices = tf.random.shuffle(indices)

shuffled_x = tf.gather(input_array, shuffled_indices)
shuffled_y = tf.gather(label_array, shuffled_indices)
"""

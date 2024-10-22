# -*- coding: utf-8 -*-
"""OptunaTransformator.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZdWR2hrAvUYuLbB5dds7noqTEshBLsfS
"""

# -*- coding: utf-8 -*-
"""SubSpaceJSOptuna.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1I9vXOXmCfmNA6dAk1ZJagTkIx6YJVkS7
"""

# -*- coding: utf-8 -*-
"""SubSpaceJS.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13TKGc6sVjhS5Ok9MTliNmm8aGU8X4xX8
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import random
import sys
import time
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from operator import add
from numpy import asarray
import math
import scipy
import seaborn as sns
#import networkx as nx
import numpy as np
#import optuna as optuna
import json
#from tabtransformertf.models.fttransformer import FTTransformerEncoder, FTTransformer
#from tabtransformertf.utils.preprocessing import df_to_dataset

####################
class Encoder(tf.keras.layers.Layer):
  def __init__(self, intermediate_dim):
    super(Encoder, self).__init__()
    self.hidden_layer = tf.keras.layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.relu,
      #kernel_initializer='he_uniform'
    )
    self.output_layer = tf.keras.layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.relu
    )
  def call(self, input_features):
    activation = self.hidden_layer(input_features)
    #print("Activation ",activation)
    return self.output_layer(activation)

class Decoder(tf.keras.layers.Layer):
  def __init__(self, intermediate_dim, original_dim):
    super(Decoder, self).__init__()
    self.hidden_layer = tf.keras.layers.Dense(
      units=intermediate_dim,
      activation=tf.nn.relu,
      #kernel_initializer='he_uniform'
    )
    self.output_layer = tf.keras.layers.Dense(
      units=original_dim,
      activation=tf.nn.relu
    )
  def call(self, code):
    activation = self.hidden_layer(code)
    return self.output_layer(activation)

class Autoencoder(tf.keras.Model):
  def __init__(self, intermediate_dim, original_dim1):
    super(Autoencoder, self).__init__()
    self.encoder1 = Encoder(intermediate_dim=intermediate_dim)
    self.decoder1 = Decoder(intermediate_dim=intermediate_dim, original_dim=original_dim1)
  
  def call(self, input_features):
    code1 = self.encoder1(input_features)
    #print("Code 1", code1)
    reconstructed1 = self.decoder1(code1)
    return reconstructed1
  def getEncoded(self, input_features):
    input_features = tf.convert_to_tensor(input_features, dtype=tf.float32)
    return self.encoder1(input_features)
 

##################### Model Creation

class Transformator(tf.keras.Model):
  def __init__(self, dim, numOfLayersE, neuronsE, activationE):
    super(Transformator, self).__init__()
    self.array=[]
    self.nL = numOfLayersE
    self.neurons = neuronsE
    self.idim = dim
    self.actF = activationE
    initializer = tf.keras.initializers.RandomUniform(minval=0.3, maxval=0.6)
    #layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)
    self.hidden_layer = tf.keras.layers.Dense(
      units=neuronsE,
      activation=activationE
      #,kernel_initializer=initializer
    )
    for i in range(self.nL):
      self.array.append(Dense(neuronsE, activation=activationE))
    self.output_layer = tf.keras.layers.Dense(
      units= self.idim,
      activation=activationE
      #,kernel_initializer=initializer
    )
  def call(self, input_features):
    encoded = self.hidden_layer(input_features)
    for i in range(self.nL):
      encoded=self.array[i](encoded)
    #print("Activation ",activation)
    return self.output_layer(encoded)
  def get_config(self):
    return {"intermediate_dim": self.idim, "numOfLayers": self.nL,"neurons": self.neurons, "activation": self.actF }
  @classmethod
  def from_config(cls, config):
      return cls(**config)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def loss(autoencoder, input_translator, source_regressor, original_target, original_target_Labels, training):
  print("Inside Loss function")
  if training == True:
    #features = np.concatenate([original_target[:,:16], original_target[:,17:]], axis = 1)
    #t1_x = np.concatenate([original_target[:,:12], original_target[:,13:]], axis = 1) for skillcraft
    """
    t1_x = np.concatenate([original_target[:,:6], original_target[:,7:]], axis = 1)
    t2_x = np.concatenate([original_target[:,:10], original_target[:,11:]], axis = 1)
    t4_x = np.concatenate([original_target[:,0:1], original_target[:,2:]], axis = 1)
    t5_x = np.concatenate([original_target[:,0:8], original_target[:,9:]], axis = 1)
    #t3_x = np.concatenate([original_target[:,:5], original_target[:,6:]], axis = 1)
    t3_x = np.concatenate([original_target[:,:4], original_target[:,5:]], axis = 1)
    
    t1_x = original_target[:,1:] # np.concatenate([original_target[:,:6], original_target[:,7:]], axis = 1)
    t2_x = np.concatenate([original_target[:,:1], original_target[:,2:]], axis = 1)
    t4_x = np.concatenate([original_target[:,0:2], original_target[:,3:]], axis = 1)
    t5_x = np.concatenate([original_target[:,0:3], original_target[:,4:]], axis = 1)
    t3_x = np.concatenate([original_target[:,:6], original_target[:,7:]], axis = 1)
    """
    #t3_x = original_target[:,:5]  #np.concatenate([original_target[:,:4], original_target[:,5:]], axis = 1)

    #features = np.concatenate([t1_x, t2_x, t3_x, t4_x, t5_x], axis = 0)
    #original_target = np.concatenate([original_target,  original_target,  original_target,  original_target, original_target], axis = 0)
    #original_target_Labels = np.concatenate([original_target_Labels,  original_target_Labels,  original_target_Labels,  original_target_Labels, original_target_Labels], axis = 0)
    
    fea = []
    og = []
    ol = []
    for feature_pointer in range(original_target.shape[1]):
        if feature_pointer == 0:
            fea.append(original_target[:, 1:original_target.shape[1]])
        elif feature_pointer == (original_target.shape[1]-1):
            fea.append(original_target[:, :feature_pointer])
        else:
            fea.append(np.concatenate([original_target[:, :feature_pointer], original_target[:,feature_pointer+1:]], axis = 1))
        og.append(original_target)
        ol.append(original_target_Labels)

    features = np.concatenate(fea, axis=0)
    original_target_Labels = np.concatenate(ol, axis=0)
    original_target = np.concatenate(og, axis=0)
    
  else:
    features = original_target

  value1 = autoencoder(features)
  sq1 = tf.reduce_mean(tf.square(tf.subtract(value1, features))) #original_target)))
  transformed = autoencoder.getEncoded(features)
  #print(f"transformed shape {transformed.shape}")
  predicted_target_labels = source_regressor.predict(transformed) #(value1)#(transformed)
  predicted_target_labels = tf.cast(predicted_target_labels, tf.float64)
  original_target_Labels = tf.cast(original_target_Labels, tf.float64)
  #transformed = tf.cast(transformed, tf.float64)
  sq1 = tf.cast(sq1, tf.float64)
  ls = tf.reduce_mean(tf.square(tf.subtract(predicted_target_labels, original_target_Labels)))
  if training == True:
    ls = 0.7*tf.reduce_mean(tf.square(tf.subtract(predicted_target_labels, original_target_Labels))) + 0.3*sq1#+0.0*tf.reduce_mean(tf.square(tf.subtract(original_target, transformed)))
      
  return ls

def transformerLoss(input_translator, source_transformer, original_target, original_target_Labels, src_numerical_features, tar_numerical_features, LABEL):
  print(f"original target {original_target}")
  transformed = input_translator(original_target.values)
  print(f"transformed {transformed}")
  print(f"transformed shape {transformed.shape}")
  transformed = np.reshape(transformed, (len(original_target), len(src_numerical_features)) )
  transformed_df = pd.DataFrame(transformed, columns = src_numerical_features)
  print(f"transformed df {transformed_df}")
  original_df = pd.concat([transformed_df, original_target_Labels], axis=1)
  print(f"Original DF {original_df}")
  origin_dataset = df_to_dataset(original_df, LABEL, shuffle=True)  
  print(f"transformed shape {transformed.shape}")
  print(f"origin dataset {origin_dataset}")
  linear_test_preds = source_transformer.predict(origin_dataset)
  linear_rms = mean_squared_error(original_target_Labels, linear_test_preds['output'].ravel(), squared=False) 
  return linear_rms

def closs(input_translator, source_regressor, original_target, original_target_Labels):
  transformed = input_translator(original_target)
  print(f"transformed shape {transformed.shape}")
  predicted_target_labels = source_regressor.predict(transformed)
  predicted_target_labels = tf.cast(predicted_target_labels, tf.float64)
  original_target_Labels = tf.cast(original_target_Labels, tf.float64)
  ls = tf.reduce_mean(tf.square(tf.subtract(predicted_target_labels, original_target_Labels)))
  return ls




def secondtrain( transformator, predictor, opt, original, original_labels):
  with tf.GradientTape() as tape, tf.GradientTape() as disc_tape:
    #tape.watch(transformator.trainable_variables)
    ls = closs(transformator, predictor, original, original_labels)
    #ls = tf.cast(ls, tf.float64)
    #ls = tf.convert_to_tensor(ls, dtype=tf.float32)
    print("secondtrain loss is ",ls)
  if (ls != 0):
    gradients = tape.gradient(ls, transformator.trainable_variables)
    gradient_variables = zip(gradients, transformator.trainable_variables)
    opt.apply_gradients(gradient_variables)
  return transformator




def train( autoencoder, transformator, predictor, opt, original, original_labels, training):
  with tf.GradientTape() as tape, tf.GradientTape() as disc_tape:
    #tape.watch(autoencoder.trainable_variables)
    #disc_tape.watch(autoencoder.trainable_variables)
    ls = loss(autoencoder, transformator, predictor, original, original_labels,training)
    ls = tf.cast(ls, tf.float64)
    #ls = tf.convert_to_tensor(ls, dtype=tf.float32)
    print("loss is ",ls)
  if (ls != 0):
    gradients = tape.gradient(ls, autoencoder.trainable_variables)
    gradient_variables = zip(gradients, autoencoder.trainable_variables)
    opt.apply_gradients(gradient_variables)
  return autoencoder


def trainwtransformer( transformator, transformer, opt, original, original_labels, src_numerical_features, tar_numerical_features, LABEL):
  with tf.GradientTape() as tape, tf.GradientTape() as disc_tape:
    tape.watch(transformator.trainable_variables)
    ls = transformerLoss(transformator, transformer, original, original_labels, src_numerical_features, tar_numerical_features, LABEL)
    #ls = tf.cast(ls, tf.float64)
    #ls = tf.convert_to_tensor(ls, dtype=tf.float32)
    print("loss is ",ls)
    if (ls != 0):
      gradients = tape.gradient(ls, transformator.trainable_variables)
      gradient_variables = zip(gradients, transformator.trainable_variables)
      opt.apply_gradients(gradient_variables)
  return transformator
#####################K fold Generation
def kfoldValidation(transformatorP, predictorP, optP, X, Y, fold, training):
  kfold = KFold(n_splits= fold, shuffle=False)
  mse_per_fold = []
  mae_per_fold = []
  mape_per_fold = []
  fold_no = 1
  #testmodel = transformatorP
  for train, test in kfold.split(X, Y):
    rowx = tf.gather(X, train)
    rowy = tf.gather(Y, train)
    #for i in range(epochS):
    #transformatorP = train(transformatorP, predictorP, optP, rowx, rowy)
    with tf.GradientTape() as tape, tf.GradientTape() as disc_tape:
      tape.watch(transformatorP.trainable_variables)
      ls = loss(transformatorP, predictorP, rowx, rowy)
      #ls = tf.cast(ls, tf.float64)
      #ls = tf.convert_to_tensor(ls, dtype=tf.float32)
      print("loss is ",ls)
      if (ls != 0):
        gradients = tape.gradient(ls, transformatorP.trainable_variables)
        gradient_variables = zip(gradients, transformatorP.trainable_variables)
        optP.apply_gradients(gradient_variables)

    tx = tf.gather(X, test)
    ty = tf.gather(Y, test)
    scores = loss(transformatorP, predictorP, tx, ty, training)
    #print(f'Score for fold {fold_no}: {testmodel.metrics_names[0]} of {scores[0]}; {testmodel.metrics_names[1]} of {scores[1]}; {testmodel.metrics_names[2]} of {scores[2]}; {testmodel.metrics_names[3]} of {scores[3]};')
    
    mse_per_fold.append(scores)
    #mae_per_fold.append(scores[2])
    #mape_per_fold.append(scores[3])
    fold_no += 1
  return transformatorP, mse_per_fold



class Objective(object):
  def __init__(self, regressor, targetP, target_labelsP, source_dim, epochs, cp_path, numFolds, training):
    # Hold this implementation specific arguments as the fields of the class.
    self.regressor = regressor
    self.org = targetP
    self.labels = target_labelsP
    self.dim = source_dim
    self.epochs = epochs
    self.savingPath = cp_path
    self.NumFolds = numFolds
    self.training = training
    
  def __call__(self, trial):
    num_layers = trial.suggest_int('num_layers', 1, 10, 2)
    neuron = trial.suggest_categorical("neuron", [10, 50, 100, 200, 300, 500, 600, 800, 900, 1000] )
    batch_size = trial.suggest_int('batch_size', low = 50, high = 300, step=50)
    momentum = trial.suggest_float('Momentum', low=0.4, high=1.0, step=0.1)
    lr2 = trial.suggest_categorical("lr2", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    optimizer = keras.optimizers.Adam(learning_rate=lr2)

    transformator  = Transformator(dim = self.dim, numOfLayersE = num_layers, neuronsE = neuron, activationE = "relu")
    losses = []
    total_loss = 0
    for epoch in range(self.epochs):
      transformator, mse = kfoldValidation(transformator, self.regressor, optimizer, self.org, self.labels, self.NumFolds, self.training)
      ls = math.sqrt(np.mean(mse))
      losses.append(ls)
      trial.report(ls, epoch)
      if trial.should_prune():
          raise optuna.exceptions.TrialPruned()
      transformator.save_weights(self.savingPath+f"Trial-{trial.number}-model")
    return ls

def finder(regressor, targetP, target_labelsP, source_dim, epochs, checkpoint_path, num_of_trials, fold, stname, storageName, training):
  tf.debugging.set_log_device_placement(True)	
  with tf.device('/GPU:0'):
    loaded_study = optuna.load_study(study_name=stname, storage=storageName)
    obj = Objective(regressor, targetP, target_labelsP, source_dim, epochs, checkpoint_path, fold, training)
    #loaded_study.optimize(obj, n_trials= num_of_trials, callbacks=[obj.callback], gc_after_trial=True)
    loaded_study.optimize( obj, n_trials= num_of_trials, gc_after_trial=True)
    trial = loaded_study.best_trial   
  return trial

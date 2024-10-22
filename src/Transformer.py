import yaml
import optuna
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten, Input, Dense, concatenate
from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Model, Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import preprocessing
import dataloader
from preprocessing import preprocessor
from dataloader import dataLoader
import sys
import os
import os.path
import csv
from sklearn.neighbors import KNeighborsRegressor
#import ensemRegressor
#import optunatransformator1
import util
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from tabtransformertf.utils.preprocessing import df_to_dataset, build_categorical_prep
from tabtransformertf.models.fttransformer import FTTransformerEncoder, FTTransformer
from be_great import GReaT
class Transformer():
  def __init__(self, yaml_name):
    self.yml_name = yaml_name
  def __call__(self, source_features, source_labels, tar_x_train, tar_y_train, tar_x_test, tar_y_test, rank):
    #clf1 = RandomForestRegressor(n_estimators=self.num_of_estimators)
    #clf1.fit(source_features, source_labels.ravel())
    
    source_features = tar_x_train
    source_labels = tar_x_test
    cols1 =[]
    cols2 = []
    for i in range(len(source_features[0])):
      cols1.append(f"C{i}")
    src_df = pd.DataFrame(source_features, columns=cols1)
    tar_df = pd.DataFrame(tar_x_test, columns= cols1)#[0:len(cols1)])
    src_l = pd.DataFrame(source_labels, columns=["TARGET"])
    #src_df["TARGET"]= source_labels.tolist()
    src_df = pd.concat([src_df, src_l], axis = 1)
    tar_df["TARGET"]= float("NAN") #np.nan
    """
    CATEGORICAL_FEATURES = []
    NUMERICAL_FEATURES = src_df.columns
    FEATURES = NUMERICAL_FEATURES
    TARGET_FEATURE = 'TARGET'
   
    #df = src_df.copy()
    #labels = src_df.pop('TARGET')
    #df = {key: value[:,tf.newaxis] for key, value in dataframe.items()}
    #df = {key: np.array(value)[:,tf.newaxis] for key, value in src_df.items()}
    #ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))

    train_dataset = df_to_dataset(src_df, TARGET_FEATURE, shuffle=True)
    ft_linear_encoder = FTTransformerEncoder(
      numerical_features=NUMERICAL_FEATURES,  # list of numeric features
      categorical_features=CATEGORICAL_FEATURES,  # list of numeric features
      numerical_data=src_df[NUMERICAL_FEATURES].values,
      categorical_data=None, # train_df[CATEGORICAL_FEATURES].values
      y = None,
      numerical_embedding_type='linear',
      embedding_dim=64,
      depth=4,
      heads=8,
      attn_dropout=0.3,
      ff_dropout=0.3,
      explainable=True
     )

    # Pass the encoder to the model
    ft_model = FTTransformer(
      encoder=ft_linear_encoder,  # Encoder from above
      out_dim=1,  # Number of outputs in final layer
      out_activation='relu',  # Activation function for final layer
      )
    epochs = 1000
    lr = 0.001
    weight_decay = 0.0001
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr
    )

    ft_model.compile(
      optimizer = optimizer,
      loss = tf.keras.losses.MeanSquaredError(),
      metrics= [tf.keras.metrics.RootMeanSquaredError()],
     )
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10)

    history = ft_model.fit(
      train_dataset,
      #ds,
      epochs=epochs,
      callbacks=[early_stopping]
      )

    dt = tar_df.copy()
    labels = tar_df.pop('TARGET')
    #df = {key: value[:,tf.newaxis] for key, value in dataframe.items()}
    dt = {key: np.array(value)[:,tf.newaxis] for key, value in tar_df.items()}
    dw = tf.data.Dataset.from_tensor_slices((dict(dt)))
    """
    #test_dataset = df_to_dataset(tar_df, shuffle=False)
    llm = GReaT(llm='distilgpt2',batch_size=32, epochs=100)
    llm.fit(src_df)
    results = llm.impute(tar_df,max_length=200)
    yhat2 = results["TARGET"] #ft_model.predict(dw)['output'].reshape(-1)
    mse3 = mean_squared_error(tar_y_test, yhat2)
    mape3 = mean_absolute_percentage_error(tar_y_test, yhat2)
    print(mse3)
    print(mape3)
    
    return mse3, mape3







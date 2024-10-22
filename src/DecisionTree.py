import yaml
#import optuna
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
class DecisionTree():
  def __init__(self, num_of_estimators):
    self.num_of_estimators = int(num_of_estimators)
  def __call__(self, source_features, source_labels, tar_x_train, tar_y_train, tar_x_test, tar_y_test, rank):
    #clf1 = RandomForestRegressor(n_estimators=self.num_of_estimators)
    #clf1.fit(source_features, source_labels.ravel())
    parameters = { 'splitter' : ['best', 'random'],
              'criterion' : ['squared_error', 'friedman_mse'],
              'max_features' : ['auto', 'sqrt'],
              'max_depth': [5, 10],
              'min_samples_split': [2, 5]
             }
    grid = GridSearchCV(DecisionTreeRegressor(),parameters)
    model = grid.fit(source_features, source_labels)
    print(model.best_params_,'\n')
    print(model.best_estimator_,'\n')    

    params = model.best_params_

    rp_source_model = DecisionTreeRegressor(**model.best_params_)
    rp_source_model.fit(tar_x_train, tar_y_train)
    yhat2 = rp_source_model.predict(tar_x_test)
    mse3 = mean_squared_error(tar_y_test, yhat2)
    mape3 = mean_absolute_percentage_error(tar_y_test, yhat2)
    print(mse3)
    print(mape3)
    
    return mse3, mape3






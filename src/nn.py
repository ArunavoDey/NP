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
#import ensemRegressor
#import optunatransformator1
import util
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


class nn():
  def __init__(self, yaml_name):
    self.yml_name = yaml_name
  def __call__(self, source_features, source_labels, tar_x_train, tar_y_train, tar_x_test, tar_y_test, rank):
    with open(os.getcwd()+self.yml_name, "r") as f:
      global_config= yaml.load(f, Loader=yaml.FullLoader)
    target_app=global_config["target_app"]
    rp_tree_model = util.sorceModelLoader(tar_x_train, tar_y_train, False, 0, True, global_config, target_app, True, os.getcwd()+global_config["source_model"], os.getcwd()+global_config["source_model_weights"] )
    yhat2 = rp_tree_model.predict(tar_x_test)
    mse3 = mean_squared_error(tar_y_test, yhat2)
    mape3 = mean_absolute_percentage_error(tar_y_test, yhat2)
    print(mse3)
    print(mape3)
    return mse3, mape3







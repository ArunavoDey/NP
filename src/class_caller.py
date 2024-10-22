import importlib
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score, mean_squared_error
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
import ensemRegressor
#import optunatransformator1
import util
from be_great import GReaT

class class_caller():
  def __init__(self, num_of_estimators):
    self.num_of_estimators = int(num_of_estimators)
  def __call__(self, config_yaml, result_path, test_samples, target_app, num_of_frozen_layers, processor,  source_features, source_labels, rank):
    with open(config_yaml, "r") as f:
      global_config= yaml.load(f, Loader=yaml.FullLoader)
    csv_path = os.getcwd()+global_config["csv_path"]
    indices_path = os.getcwd()+global_config["indices_path"]
    #callback2 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=40)
    folder_name = "class_caller"
    list_of_classes = global_config["class_caller"]["list_of_classes"]
    list_of_class_args = global_config["class_caller"]["list_of_class_args"]
    llm = GReaT(llm='distilgpt2', batch_size=32, epochs=100)
    os.environ["WANDB_DISABLED"] = "true"
    tar_x_train, tar_x_test, tar_y_train, tar_y_test = processor.get_train_test_target(test_size = 0.9, rand_state=rank*50)
    cols1 =[]
    cols2 = []
    for i in range(len(source_features[0])):
      cols1.append(f"C{i}")
    src_df = pd.DataFrame(source_features, columns=cols1)
    tar_df = pd.DataFrame(tar_x_test, columns= cols1[0:len(cols1)-1])
    tar_df[cols1[len(cols1)-1]]= float("NAN") #np.nan
    llm.fit(src_df)
    new_tar_x_test = llm.impute(tar_df, max_length=400)

    for i in range(rank, rank+1):
      os.makedirs(os.path.dirname(f"{csv_path}{folder_name}/Source-model-on-target-{target_app}-{folder_name}-results-{rank}-MSE.csv"), exist_ok=True)
      os.makedirs(os.path.dirname(f"{csv_path}{folder_name}/Source-model-on-target-{target_app}-{folder_name}-results-{rank}-MAPE.csv"), exist_ok=True)
      fileI = open(f"{csv_path}{folder_name}/Source-model-on-target-{target_app}-{folder_name}-results-{rank}-MSE.csv", "w")
      fileJ = open(f"{csv_path}{folder_name}/Source-model-on-target-{target_app}-{folder_name}-results-{rank}-MAPE.csv", "w")
      writerI = csv.writer(fileI)
      writerJ = csv.writer(fileJ)
      class_number = 0
      for module_name in list_of_classes:
        module = importlib.import_module(module_name)
        func = getattr(module, module_name)
        obj = func(list_of_class_args[class_number])
        mse0, mape0 = obj(source_features, source_labels, new_tar_x_test, tar_y_test, rank)
        rowMSE =[]
        rowMAPE = []
        rowMSE.append(module_name)
        rowMSE.append(mse0)
        writerI.writerow(rowMSE)
        rowMAPE.append(module_name)
        rowMAPE.append(mape0)
        writerJ.writerow(rowMAPE)
        class_number = class_number + 1
      fileI.close()
      fileJ.close()


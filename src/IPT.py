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
from tensorflow.keras.models import Model, Sequential, load_model, model_from_json
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
import optunatransformator1
import util


class IPT():
  def __init__(self, num ):
    self.num = int(num)
  def __call__(self, config_yaml, result_path, test_samples, target_app, num_of_frozen_layers, processor,  source_features, source_labels, rank):
    with open(config_yaml, "r") as f:
      global_config= yaml.load(f, Loader=yaml.FullLoader)
    csv_path = os.getcwd()+global_config["csv_path"]
    indices_path = os.getcwd()+global_config["indices_path"]
    use_case_specific_config = global_config["IPT"]["IPT_params"]
    folder_name = "IPT"
    #rank = rank + 1
    for i in range(rank, rank+1):
      os.makedirs(os.path.dirname(f"{csv_path}{folder_name}/Source-model-on-target-{target_app}-{folder_name}-results-{rank}.csv"), exist_ok=True)
      fileI = open(f"{csv_path}{folder_name}/Source-model-on-target-{target_app}-{folder_name}-results-{rank}.csv", "w")
      fileI2 = open(f"{csv_path}{folder_name}/Source-model-on-target-{target_app}-MAPE-{folder_name}-results-{rank}.csv", "w")
      writer = csv.writer(fileI)
      writer2 = csv.writer(fileI2)
      tar_x_train, tar_x_test, tar_y_train, tar_y_test = processor.get_train_test_target(test_size = 0.9, rand_state=i*50)
      for j in test_samples:
        if isinstance(j, int):
          fname = f"{j}-samples"
          fidx = j/5
        else:
          fname = f"{j}-percent"
          fidx = int(j*10.0)
        n = j/10
        tar_x_scaled, tar_y_scaled = processor.get_tar_train()
        x2, lb2, tar_x_scaled, tar_y_scaled = util.sampleLoader(tar_x_scaled, tar_y_scaled,f"{indices_path}/{target_app}-indices-{rank}-{fname}.csv" ,j)
        print(f"just loaded x2 {x2} for {fname}")
        rowArr = []
        rowMape = []
        source_model = util.sourceModelLoader(source_features, source_labels, False, num_of_frozen_layers, False, global_config, target_app, False)
        callback2 = util.EarlyStoppingAtMinLoss(patience=40, arg_loss="loss")
        transformator = optunatransformator1.Transformator(dim = source_features.shape[1], numOfLayersE = use_case_specific_config["layers"] , neuronsE = use_case_specific_config["neurons"] , activationE= "relu")
        optimizer = tf.keras.optimizers.Adam(learning_rate= use_case_specific_config["lr"])
        for epch in range(1000):
          print(f"x2 {x2}")
          print(f"lb2 {lb2}")
          optunatransformator1.train( transformator, source_model, optimizer, x2, lb2)
        """
        predictions0 = source_model.predict(tar_x_test)
        pp0 = np.nan_to_num(predictions0)# tf.cast(predictions0, dtype = tf.float32)
        mse0 = mean_squared_error(tar_y_test, pp0)
        rowArr.append(mse0)
        """
        transformed = transformator(tar_x_test)
        predictions1 = source_model.predict(transformed)
        pp1 = np.nan_to_num(predictions1)# tf.cast(predictions0, dtype = tf.float32)
        mse1 = mean_squared_error(tar_y_test, pp1)
        mape1 = mean_absolute_percentage_error(tar_y_test, pp1)
        rowArr.append(mse1)
        rowMape.append(mape1)
        writer.writerow(rowArr)
        writer2.writerow(rowMape)
      fileI.close()
      fileI2.close()

	

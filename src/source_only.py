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




class source_only():
  def __init__(self, value):
    self.value = int(value)
  def __call__(self, config_yaml, result_path, test_samples, target_app, num_of_frozen_layers, processor, source_features, source_labels, rank):
    with open(config_yaml, "r") as f:
      global_config= yaml.load(f, Loader=yaml.FullLoader)
    fig_path = os.getcwd()+global_config["fig_path"]
    _ , tar_x_scaled, _, tar_y_scaled = processor.get_train_test_target(test_size=0.9)
    source_test_features, source_test_labels = processor.getSrcTestSamples()
    source_model = util.sourceModelLoader(source_features, source_labels, False, num_of_frozen_layers, True, global_config, target_app, True)
    f.close()
    """# testing model on Source data"""
    predictions = source_model.predict(source_test_features)
    #util.yaml_key_adder(f"{target_app}-config.yaml","source_model",f"{os.getcwd()+global_config['model_path']}{target_app}/{target_app}-model.json" )
    #util.yaml_key_adder(f"{target_app}-config.yaml","source_model_weights",f"{os.getcwd()+global_config['model_path']}{target_app}/{target_app}-model")
    util.scatterPlot(source_test_labels, predictions,f"{fig_path}Testing-Source-model-on-source-{target_app}-{rank}.pdf", f"source-on-source-{target_app}")

    """#testing model on target data"""
    tar_pred_scaled = source_model.predict(tar_x_scaled)
    util.scatterPlot(tar_y_scaled, tar_pred_scaled,f"{fig_path}Testing-Source-model-on-target-{target_app}-{rank}.pdf", f"source-on-target-{target_app}")
    #util.plot_graphs(history, "loss", f"{result_path}figs/Source-model-on-source-{target_app}-train-val-loss.pdf")


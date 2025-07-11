import argparse

neural_decoding_dir = "D:/ND/neuraldecoding"
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
args = parser.parse_args()

cfg_path = args.config

import sys
import os
sys.path.append(neural_decoding_dir)

import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pickle

from hydra import initialize, compose

from neuraldecoding.decoder import NeuralNetworkDecoder, LinearDecoder
from neuraldecoding.trainer.LinearTrainer import LinearTrainer
from neuraldecoding.utils import load_one_nwb, parse_verify_config, prep_data_decoder, accuracy
from neuraldecoding.preprocessing import Preprocessing

from hydra import initialize, compose

with initialize(version_base=None, config_path=cfg_path):
    cfg = compose("config")

decoder_config = parse_verify_config(cfg, 'decoder')
trainer_config = parse_verify_config(cfg, 'trainer')
preprocessing_config = parse_verify_config(cfg, 'preprocessing')
preprocessing_trainer_config = preprocessing_config['preprocessing_trainer']
preprocessing_decoder_config = preprocessing_config['preprocessing_decoder']

print("Starting trainer preprocessing...")
preprocessor_trainer = Preprocessing(preprocessing_trainer_config)

trainer = LinearTrainer(preprocessor_trainer, trainer_config)

# train
model, results = trainer.train_model()

model_path = decoder_config["fpath"]
model.save_model(decoder_config["fpath"])

# decode
preprocessor_decoder = Preprocessing(preprocessing_decoder_config)
decoder = LinearDecoder(decoder_config)
decoder.load_model()


data_path = trainer_config['data']['data_path']
with open(data_path, "rb") as f:
    data_dict = pickle.load(f)

neural_test, finger_test  = preprocessor_decoder.preprocess_pipeline(data_dict, params = {'is_train': False})
print("Starting prediction...")
prediction = decoder.predict(neural_test)

print(f"Prediction: {prediction}, Accuracy: {accuracy(prediction, finger_test)}")

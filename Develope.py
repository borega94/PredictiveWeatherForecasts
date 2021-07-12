"""
Project: Predictive Weather Forecasts
Name: Philipp Kataliako
Date: 20.05.2021
Description: 
Dieses Skript soll die bereinigten und aufbereiteten Daten in Timeseries format umwandeln 
"""
from data_prepare import dataPrepare
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from data_to_timeseries import toTimeseries
from plotten import boxplots
from Neuronale_Netze import neuronal_network
from Neuronale_Netze import neuronal_network_keras
from data_splitten import split_data
<<<<<<< Updated upstream
#import numpy as np
=======
from Normalization import normalization
import numpy as np
>>>>>>> Stashed changes
import pandas as pd

INPUT = pd.read_csv('produkt_wetter_tag_19490101_20140131_01346.txt', sep=";")

x_set, y_set, data_set = toTimeseries(INPUT)



X_train, y_train, X_test, y_test, X_val, y_val = split_data(x_set, y_set)

<<<<<<< Updated upstream
#neuronal_network(X_train, y_train, X_test, y_test, X_val, y_val)
=======
neuronal_network(X_train, y_train, X_test, y_test, X_val, y_val)


>>>>>>> Stashed changes

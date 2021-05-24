"""
Project: Predictive Weather Forecasts
Name: Philipp Kataliako
Date: 20.05.2021
Description: 
Dieses Skript soll die bereinigten und aufbereiteten Daten in Timeseries format umwandeln 
"""
from data_prepare import dataPrepare
from data_to_timeseries import toTimeseries
from plotten import boxplots
from Neuronale_Netze import neuronal_network
from data_splitten import split_data
import numpy as np
import pandas as pd

INPUT = pd.read_csv('produkt_wetter_tag_19490101_20140131_01346.txt', sep=";")

x_set, y_set, data_set= toTimeseries(INPUT)

#print(x_set.head(3))
#print(y_set.head(3))

#boxplots(data_set)


X_train, y_train, X_test, y_test, X_val, y_val = split_data(x_set, y_set)

neuronal_network(X_train, y_train, X_test, y_test, X_val, y_val)

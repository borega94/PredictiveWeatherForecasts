"""
Project: Predictive Weather Forecasts
Name: Philipp Kataliako
Date: 20.05.2021
Description: 
Dieses Skript soll die bereinigten und aufbereiteten Daten in Timeseries format umwandeln 
"""
from data_prepare import dataPrepare
from data_to_timeseries import toTimeseries
import numpy as np
import pandas as pd

INPUT = pd.read_csv('produkt_wetter_tag_19490101_20140131_01346.txt', sep=";")

x_set, y_set = toTimeseries(INPUT)

print(x_set.head(3))
print(y_set.head(3))
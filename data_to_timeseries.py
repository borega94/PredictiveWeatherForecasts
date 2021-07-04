"""
Project: Predictive Weather Forecasts
Name: Philipp Kataliako
Date: 20.05.2021
Description: 
Dieses Skript soll die bereinigten und aufbereiteten Daten in Timeseries format umwandeln.

"""
from data_prepare import dataPrepare
import numpy as np
import pandas as pd


"""
Diese Funktion nimmt die Daten CSV File als Input und gibt das X Datenset mit den Trainingsdaten und Y Datenset mit den Labels aus
"""
def toTimeseries(File = None):

    data = dataPrepare(File)
    data.reset_index(level=0, inplace=True)

    #data.drop(columns=["Hagel"], inplace=True)
    #print(data)
    #input()

    # Erstellen von drei gleichen Dataframes, Umbennen der Spalten
    tag_1 = data
    tag_2 = data
    tag_3 = data

    tag_1 = tag_1.add_prefix("Tag1-")
    tag_2 = tag_2.add_prefix("Tag2-")
    tag_3 = tag_3.add_prefix("Tag3-")

    # Droppen der ersten bzw. der ersten beiden Tage, Zurücksetzten der Index
    tag_2 = tag_2.iloc[1:]
    tag_3 = tag_3.iloc[2:]

    tag_2.reset_index(drop= True, inplace=True)
    tag_3.reset_index(drop=True, inplace=True)
    # Jeweils der dritte Tag der Reihe behält die Hagel Spalte
    tag_1.drop(columns=["Tag1-Hagel"], inplace=True)
    tag_2.drop(columns=["Tag2-Hagel"], inplace=True)

    # Dataframes horizontal anhängen 
    tage_312 = pd.concat([tag_3, tag_1, tag_2], axis=1)
    #tage_312 = pd.concat([tage_31, tag_2], axis=1)

    # Entfernen der unnötigen Spalten, Löschen der letzten 2 Spalten
    tage_312.drop(columns=["Tag1-MESS_DATUM", "Tag2-MESS_DATUM"], inplace=True)
    tage_312 = tage_312.iloc[:-2]

    # Mache Datumspalte wieder index
    tage_312.set_index('Tag3-MESS_DATUM', inplace=True)

    x_dataset = tage_312.drop(columns="Tag3-Hagel")
    y_dataset = tage_312["Tag3-Hagel"]

    return x_dataset, y_dataset, data

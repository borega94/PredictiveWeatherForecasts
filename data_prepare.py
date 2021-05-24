"""
Project: Predictive Weather Forecasts
Name: Philipp Kataliako
Date: 20.05.2021
Description: 
Dieses Skript basiert auf dem Skript "Feldberg_Analyse_WetterAnHageltagen". 
Die nötigen Schritte zur Datenbereinigung werden hier zusammengefasst und als Python Funktion zum Abruf bereitgestellt. 
"""

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

#INPUT = pd.read_csv('produkt_wetter_tag_19490101_20140131_01346.txt', sep=";")

def dataPrepare(input_data = None):
    
    feldbergHageltage = input_data

    # Mess_Datum in Datentyp Datetime umwandeln und als Index setzen
    #feldbergHageltage['MESS_DATUM'] = pd.to_datetime(arg=feldbergHageltage['MESS_DATUM'], format='%Y%m%d')
    feldbergHageltage.set_index('MESS_DATUM', inplace=True)

    feldbergWetter = pd.read_csv('produkt_klima_tag_19450101_20201231_01346.txt', sep=";")
    #feldbergWetter['MESS_DATUM'] = pd.to_datetime(arg=feldbergWetter['MESS_DATUM'], format='%Y%m%d')
    feldbergWetter.set_index('MESS_DATUM', inplace=True)

    # feldbergHageltage enthält nur noch Index und die Spalte HAGEL
    feldbergHageltage = feldbergHageltage['HAGEL']
    # Spalte HAGEL in feldbergWetter angehängt
    feldbergWetter['HAGEL'] = feldbergHageltage

    # Kontrolle welche Werte in der Spalte HAGEL auftreten = 0, 1, 99
    #feldbergWetter['HAGEL'].unique()

    # Löscht alle Zeilen in den irgendwo nan steht - löscht möglicherweise auch Zeilen die nicht gelöscht werden sollen
    #print(feldbergWetter.dropna())

    # Löscht alle Zeilen in der Spalte HAGEL in denen nan steht
    feldbergWetter = feldbergWetter[feldbergWetter['HAGEL'].notna()]

    # Löschen der Spalten die irrelevant sind
    feldbergWetter = feldbergWetter.drop(columns=['QN_3', 'QN_4', 'eor'])

    # Umbenennung der Spaltennamen
    feldbergWetter = feldbergWetter.rename(columns={'  FX': 'WindstaerkeMax','  FM': 'WindstaerkeMittel',
                                    ' RSK': 'Niederschlagshoehe', 'RSKF': 'Niederschlagsform', ' SDK': 'Sonnenscheindauer',
                                    'SHK_TAG': 'Schneehoehe', '  NM': 'Bewoelkung', ' VPM': 'DampfdruckMittel',
                                    '  PM': 'LuftdruckMittel', ' TMK': 'TemperaturMittel', ' UPM': 'RelativeFeuchteMittel',
                                    ' TXK': 'LufttemperaturMax', ' TNK': 'LufttemperaturMin2m', ' TGK': 'LufttemperaturMin5cm',
                                    'HAGEL': 'Hagel'})

    # ---------------------------------------------------------------------------------------
    # Zwischenstand : Datei mit Werten in Dataframe umgewandelt und Spalten eindeutig benannt.
    # Hagel in der Tabelle mit dem Wetter angehängt
    # Zeilen gelöscht in denen kein Wert für Hagel vorhanden ist
    # ---------------------------------------------------------------------------------------

    # Anzeigen der Spalten
    # print(feldbergWetter.columns)

    # Anzeigen aller Spalten bei einer Ausgabe mit 'print'
    pd.set_option('max_columns', None)
    #print(feldbergWetter.head(20))


    # -999.0 Werte in jeder Spalte zählen / -999.0 = Fehlwert
    fehlwerteInSpalten = {
        'fehlwerteWindstaerkeMax' : feldbergWetter.WindstaerkeMax==-999.0,
        'fehlwerteWindstaerkeMittel' : feldbergWetter.WindstaerkeMittel==-999.0,
        'fehlwerteNiederschlagshoehe' : feldbergWetter.Niederschlagshoehe==-999.0,
        'fehlwerteNiederschlagsform' : feldbergWetter.Niederschlagsform==-999.0,
        'fehlwerteSonnenscheindauer' : feldbergWetter.Sonnenscheindauer==-999.0,
        'fehlwerteSchneehoehe' : feldbergWetter.Schneehoehe==-999.0,
        'fehlwerteBewoelkung' : feldbergWetter.Bewoelkung==-999.0,
        'fehlwerteDampfdruckMittel' : feldbergWetter.DampfdruckMittel==-999.0,
        'fehlwerteLuftdruckMittel' : feldbergWetter.LuftdruckMittel==-999.0,
        'fehlwerteTemperaturMittel' : feldbergWetter.TemperaturMittel==-999.0,
        'fehlwerteRelativeFeuchteMittel' : feldbergWetter.RelativeFeuchteMittel==-999.0,
        'fehlwerteLufttemperaturMax' : feldbergWetter.LufttemperaturMax==-999.0,
        'fehlwerteLufttemperaturMin2m' : feldbergWetter.LufttemperaturMin2m==-999.0,
        'fehlwerteLufttemperaturMin5cm' : feldbergWetter.LufttemperaturMin5cm==-999.0,
        'fehlwerteHagel' : feldbergWetter.Hagel==-999.0
    }
    df_Fehlwerte = pd.DataFrame(fehlwerteInSpalten)
    # Liefert die Anzahl der Fehlwerte in jeder Spalte
    #print(df_Fehlwerte.sum())

    # Spalte WindstaerkeMittel aufgrund zu vieler Fehlwerte droppen
    #feldbergWetter = feldbergWetter.drop(columns=['WindstaerkeMittel'])

    # Test ob ein Zusammenhang zwischen den Spalten zu erkennen ist
    #print(feldbergWetter.corr())

    # Droppen der Spalte Windstaerke Mittel, aufgrund zuvieler Fehlwerte
    feldbergWetter = feldbergWetter.drop(columns=['WindstaerkeMittel'])

    feldbergWetter.drop(columns=["STATIONS_ID"], inplace=True)
    feldbergWetter.drop(columns=["Niederschlagsform"], inplace=True)
    feldbergWetter.drop(columns=["DampfdruckMittel"], inplace=True)
    feldbergWetter.drop(columns=["TemperaturMittel"], inplace=True)
    feldbergWetter.drop(columns=["LufttemperaturMin5cm"], inplace=True)
    feldbergWetter.drop(columns=["RelativeFeuchteMittel"], inplace=True)
    feldbergWetter.drop(columns=["LuftdruckMittel"], inplace=True)


    # Löschen aller Zeilen bis zum 1.1.1955
    feldbergWetter.drop(feldbergWetter.loc['1949-01-01':'1954-12-31'].index, inplace=True)

    # Fehlwerte ersetzen
    feldbergWetter.replace(-999.0, np.nan, inplace=True)
    # NaN-Werte werden durch linear interpolierte Werte, mit dem Wert vor und nach der Spalte ersetzt
    feldbergWetter.interpolate(inplace=True)


        
    return feldbergWetter

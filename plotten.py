"""
Project: Predictive Weather Forecasts
Name: Carsten
Date: 24.05.2021
Description:
Dieses Skript soll die Daten auf verschiedene Art und Weisen visualisieren
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def boxplots(input_df):
    df_visu = input_df

    fig, ax = plt.subplots(2, 6)
    sns.boxplot(x='Hagel', y='WindstaerkeMax', data=df_visu, ax=ax[0, 0])
    sns.boxplot(x='Hagel', y='Niederschlagshoehe', data=df_visu, ax=ax[0, 1])
    sns.boxplot(x='Hagel', y='Niederschlagsform', data=df_visu, ax=ax[0, 2])
    sns.boxplot(x='Hagel', y='Sonnenscheindauer', data=df_visu, ax=ax[0, 3])
    sns.boxplot(x='Hagel', y='Schneehoehe', data=df_visu, ax=ax[0, 4])
    sns.boxplot(x='Hagel', y='Bewoelkung', data=df_visu, ax=ax[0, 5])
    sns.boxplot(x='Hagel', y='DampfdruckMittel', data=df_visu, ax=ax[1, 0])
    sns.boxplot(x='Hagel', y='LuftdruckMittel', data=df_visu, ax=ax[1, 1])
    sns.boxplot(x='Hagel', y='TemperaturMittel', data=df_visu, ax=ax[1, 2])
    sns.boxplot(x='Hagel', y='RelativeFeuchteMittel', data=df_visu, ax=ax[1, 3])
    sns.boxplot(x='Hagel', y='LufttemperaturMax', data=df_visu, ax=ax[1, 4])
    sns.boxplot(x='Hagel', y='LufttemperaturMin2m', data=df_visu, ax=ax[1, 5])
    plt.show()

    return
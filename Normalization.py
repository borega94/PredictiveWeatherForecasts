"""
Project: Predictive Weather Forecasts
Name: Alexandra, Carsten
Date: 24.06.2021
Description:
Dieses Skript soll unsere Wetterdaten normalisieren
"""

def normalization (dataset):

    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd
    scaler = MinMaxScaler()

    min_heart = dataset.min()
    max_heart = dataset.max()

    dataset_normalized = scaler.fit_transform(dataset)

    dataset_normalized = pd.DataFrame(dataset_normalized)

    dataset_normalized.columns = dataset.columns



    return dataset_normalized


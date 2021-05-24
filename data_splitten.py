"""
Project: Predictive Weather Forecasts
Name: Alexandra, Carsten
Date: 24.05.2021
Description:
Dieses Skript soll die bereinigten und aufbereiteten Daten in Trainings und Validierungsdaten splitten
"""

from sklearn.model_selection import train_test_split

def split_data(input_x, input_y):

    X = input_x
    y = input_y

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, shuffle=False)

    # take the remaining 20% of data in X_tmp, y_tmp and split them evenly
    #X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=23)
    X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, shuffle=False)

    return X_train, y_train, X_test, y_test, X_val, y_val
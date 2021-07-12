import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import explained_variance_score, \
    mean_absolute_error, \
    median_absolute_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.utils.vis_utils import plot_model
from data_to_timeseries import toTimeseries
from data_splitten import split_data
from accuracy_score_altered import accuracy_score as accuracy_score_alt

alt = True


INPUT = pd.read_csv('produkt_wetter_tag_19490101_20140131_01346.txt', sep=";")

x_set, y_set, data_set = toTimeseries(INPUT)

X_train, y_train, X_test, y_test, X_val, y_val = split_data(x_set, y_set)

feature_cols = [tf.feature_column.numeric_column(col) for col in X_train.columns]
regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[3, 32],
                                          model_dir='tf_wx_model',
                                          label_dimension=1)


def wx_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=1024):
    return tf.compat.v1.estimator.inputs.pandas_input_fn(x=X,
                                                         y=y,
                                                         num_epochs=num_epochs,
                                                         shuffle=shuffle,
                                                         batch_size=batch_size)


predictions = regressor.predict(input_fn=wx_input_fn(X_test, y_test, num_epochs=1, shuffle=False))

pred = np.array([p['predictions'][0] for p in predictions])

ergebnis = pd.DataFrame(columns=['Tag', 'Hagel', 'Vorhersage'])
ergebnis['Tag'] = X_test.index
ergebnis.set_index('Tag', inplace=True)
ergebnis['Hagel']=y_test
ergebnis['Vorhersage'] = pred

print('Hagelvorhersage > 0.7: ')
print(ergebnis[ergebnis['Vorhersage'] > 0.7])

print('Tage an denen es gehagelt hat und die Vorhersage größer 0,5 war: ')
print(ergebnis[(ergebnis['Vorhersage'] > 0.7) & (ergebnis['Hagel']==1.0)])
print('Anzahl der Hageltage im Datensatz: ', ergebnis['Hagel'].sum())

# Vergleich zwischen vorhergesagten und tatsächlichen Hagelwerten
result = np.where(pred<0.7)
gtruth = np.where(y_test==1)

print('Result ' , result)
print('gtruth ' , gtruth)

# Accuracy Score aufsetzen
acc_Gruth = pd.Series(y_test)
acc_Pred = pd.Series(predictions)

acc_Gruth.reset_index(drop = True)

# Rauslöschen der Reihen bei denen beide Datensätze "kein Hagel" haben
manuel_index = 0
anzahl_gelöscht = 0

lösch_index = []
"""
if not alt:
    for i, rows in acc_Gruth.iteritems():
        if rows == -1 & acc_Pred.iloc[manuel_index]==-1:
            lösch_index.append(manuel_index)
            anzahl_gelöscht += 1
        manuel_index = manuel_index + 1

    acc_Gruth = acc_Gruth.drop(acc_Gruth.index[lösch_index])
    acc_Pred = acc_Pred.drop(acc_Pred.index[lösch_index])
"""
# Berechnen der Score
if alt:
    accuracy_score = accuracy_score_alt
accuracy_prozent = accuracy_score(acc_Gruth, acc_Pred, normalize = True)
accuracy_absolut = accuracy_score(acc_Gruth, acc_Pred, normalize = False)

print("Prozentual richtig: ", accuracy_prozent)
print("Absolut richtig: ", accuracy_absolut)





"""
Project: Predictive Weather Forecasts
Name: Philipp, Judith, Christof
Date: 20.05.2021
Description: 
Dieses Skript soll den Decision Tree aufsetzen und evaluieren
"""
from numpy.core.arrayprint import printoptions
from numpy.lib.function_base import gradient
from sklearn import tree
from sklearn.metrics import accuracy_score
from data_prepare import dataPrepare
from data_to_timeseries import toTimeseries
from plotten import boxplots
from Neuronale_Netze import neuronal_network
from data_splitten import split_data_simple
import numpy as np
import pandas as pd
import graphviz

vis = False

# Daten importieren
INPUT = pd.read_csv('produkt_wetter_tag_19490101_20140131_01346.txt', sep=";")

# Daten aufbereiten und in Timeseries umwandeln
x_set, y_set, data_set= toTimeseries(INPUT)

# Daten in Training- und Testset splitten
X_train, X_test, Y_train, Y_test = split_data_simple(x_set, y_set)

# Umwandeln der Labels in integer und ersetzten von 0 mit -1
Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

Y_train.replace(0, -1, inplace = True)
Y_test.replace(0, -1, inplace = True)

# Decistion Tree fitten
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

# Vorhersage erstellen
prediciton = clf.predict(X_test)

# Vergleich zwischen vorhergesagten und tatsächlichen Hagelwerten
result = np.where(prediciton==1)
gtruth = np.where(Y_test==1)
#print(result[0], " gruth: ", gtruth[0])

# Accuracy Score aufsetzen
acc_Gruth = pd.Series(Y_test)
acc_Pred = pd.Series(prediciton)

acc_Gruth.reset_index(drop = True)

# Rauslöschen der Reihen bei denen beide Datensätze "kein Hagel" haben
manuel_index = 0
anzahl_gelöscht = 0

lösch_index = []

for i, rows in acc_Gruth.iteritems():    
    if rows == -1 & acc_Pred.iloc[manuel_index]==-1:
        lösch_index.append(manuel_index)
        anzahl_gelöscht += 1
    manuel_index = manuel_index + 1

print(lösch_index)
print(anzahl_gelöscht)
print(acc_Gruth)
print(acc_Pred)

reihen = acc_Gruth.index[lösch_index]
acc_Gruth = acc_Gruth.drop(acc_Gruth.index[lösch_index])
acc_Pred = acc_Pred.drop(acc_Pred.index[lösch_index])


print(acc_Gruth)
print(acc_Pred)

accuracy_prozent = accuracy_score(acc_Gruth, acc_Pred, normalize = True)
accuracy_absolut = accuracy_score(acc_Gruth, acc_Pred, normalize = False)

print("Prozentual richtig: ", accuracy_prozent)
print("Absolut richtig: ", accuracy_absolut)

if vis:
    # Visualisierung des Trees
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=X_train.columns, class_names=["Hagel", "Kein Hagel"], filled=True, rounded =True)
    graph = graphviz.Source(dot_data)
    graph.render("DecisionTree")

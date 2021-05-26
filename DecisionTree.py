"""
Project: Predictive Weather Forecasts
Name: Philipp, Judith, Christof
Date: 20.05.2021
Description: 
Dieses Skript soll den Decision Tree aufsetzen
"""
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

INPUT = pd.read_csv('produkt_wetter_tag_19490101_20140131_01346.txt', sep=";")

x_set, y_set, data_set= toTimeseries(INPUT)

X_train, X_test, Y_train, Y_test = split_data_simple(x_set, y_set)

Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

Y_train.replace(0, -1, inplace = True)
Y_test.replace(0, -1, inplace = True)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

prediciton = clf.predict(X_test)

# Vergleich zwischen vorhergesagten und tatsächlichen Hagelwerten
result = np.where(prediciton==1)
gtruth = np.where(Y_test==1)
print(result[0], " gruth: ", gtruth[0])

#accuracy_prozent = accuracy_score(gtruth, result, normalize = True)
#accuracy_absolut = accuracy_score(gtruth, result, normalize = False)

#print("Prozentual richtig: ", accuracy_prozent)
#print("Absolut richtig: ", accuracy_absolut)

# Visualisierung des Trees
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=X_train.columns, class_names=["Hagel", "Kein Hagel"], filled=True, rounded =True)
graph = graphviz.Source(dot_data)
graph.render("DecisionTree")

tree.plot_tree(clf, max_depth=10) 

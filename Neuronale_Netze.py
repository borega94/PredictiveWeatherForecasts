"""
Project: Predictive Weather Forecasts
Name: Alexandra, Carsten
Date: 24.05.2021
Description:
Dieses Skript soll die bereinigten und aufbereiteten Daten mithilfe eines Neuronalen Netzes modellieren
"""

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

def neuronal_network_test(X_train, y_train, X_test, y_test, X_val, y_val):



    feature_cols = [tf.feature_column.numeric_column(col) for col in X_train.columns]

    regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[50, 50],
                                          model_dir='tf_wx_model')

    def wx_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=400):
        return tf.compat.v1.estimator.inputs.pandas_input_fn(x=X,
                                                             y=y,
                                                             num_epochs=num_epochs,
                                                             shuffle=shuffle,
                                                             batch_size=batch_size)

    regressor.train(input_fn=wx_input_fn(X_train, y_train, num_epochs=None, shuffle=True), steps=1)
    regressor.evaluate(input_fn=wx_input_fn(X_val, y_val, num_epochs=1, shuffle=False), steps=1)

    predictions = regressor.predict(input_fn=wx_input_fn(X_test, y_test, num_epochs=1, shuffle=False))


    return

def neuronal_network(X_train, y_train, X_test, y_test, X_val, y_val):

    feature_cols = [tf.feature_column.numeric_column(col) for col in X_train.columns]

    regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                          hidden_units=[50, 50],
                                          model_dir='tf_wx_model',
                                          label_dimension=1)

    def wx_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=4):
        return tf.compat.v1.estimator.inputs.pandas_input_fn(x=X,
                                                             y=y,
                                                             num_epochs=num_epochs,
                                                             shuffle=shuffle,
                                                             batch_size=batch_size)

    evaluations = []
    STEPS = 40

    for i in range(10):
        regressor.train(input_fn=wx_input_fn(X_train, y_train, num_epochs=None, shuffle=True), steps=STEPS)
        evaluations.append(regressor.evaluate(input_fn=wx_input_fn(X_val,
                                                                   y_val,
                                                                   num_epochs=1,
                                                                   shuffle=False)))
        print(evaluations[-1])

    print(evaluations[-1])

    predictions = regressor.predict(input_fn=wx_input_fn(X_test, y_test, num_epochs=1, shuffle=False))



    # manually set the parameters of the figure to and appropriate size
    plt.rcParams['figure.figsize'] = [14, 10]

    loss_values = [ev['loss'] for ev in evaluations]
    training_steps = [ev['global_step'] for ev in evaluations]

    plt.scatter(x=training_steps, y=loss_values)
    plt.xlabel('Training steps (Epochs = steps / 2)')
    plt.ylabel('Loss (SSE)')
    plt.show()
    #pred = np.array([p['predictions'][0] for p in predictions])
    return
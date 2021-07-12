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
from tensorflow import keras
from keras.utils.vis_utils import plot_model

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
                                          hidden_units=[3, 32],
                                          model_dir='tf_wx_model',
                                          label_dimension=1)

    def wx_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=1024):
        return tf.compat.v1.estimator.inputs.pandas_input_fn(x=X,
                                                             y=y,
                                                             num_epochs=num_epochs,
                                                             shuffle=shuffle,
                                                             batch_size=batch_size)

    evaluations = []
    STEPS = 12

    for i in range(48):
        regressor.train(input_fn=wx_input_fn(X_train, y_train, num_epochs=None, shuffle=True), steps=STEPS)
        evaluations.append(regressor.evaluate(input_fn=wx_input_fn(X_val,
                                                                   y_val,
                                                                   num_epochs=1,
                                                                   shuffle=False)))
        print(evaluations[-1])
        print(i)

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

    pred = np.array([p['predictions'][0] for p in predictions])

    ergebnis = pd.DataFrame(columns=['Tag', 'Hagel', 'Vorhersage'])
    ergebnis['Tag'] = X_test.index
    ergebnis.set_index('Tag', inplace=True)
    ergebnis['Hagel']=y_test
    ergebnis['Vorhersage'] = pred
    #print(y_test)
    #print(ergebnis.info)
    print('Tage an denen es gehagelt hat und die Vorhersage größer 0,5 war: ')
    print(ergebnis[(ergebnis['Vorhersage'] > 0.5) & (ergebnis['Hagel']==1.0)])
    print('Anzahl der Hageltage im Datensatz: ', ergebnis['Hagel'].sum())
    print('Hagelvorhersage > 0.5: ')
    print(ergebnis[ergebnis['Vorhersage']>0.5])
    """
    counterFunc = ergebnis.apply(
        lambda x: True if x[2] > 0.5 else False, axis=2)
    numOfRows = len(counterFunc[counterFunc == True].index)
    print('Tage an denen Hagelvorhersage > 0.5: ', numOfRows)"""


    return


def neuronal_network_keras(X_train, y_train, X_test, y_test):

    """ Stop des Trainieres bei entsprechender Genauigkeit"""
    val_acc_threshold = 0.98
    acc_threshold = 0.98

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if ((logs.get('accuracy') > acc_threshold) & (logs.get('val_accuracy') > val_acc_threshold)):
                print("Reached  val_accuracy, so stopping training!!".format(acc_threshold))
                self.model.stop_training = True


    callbacks = myCallback()


    """ Hülle unseres Modells """
    model = keras.Sequential([
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    adam_opt = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=adam_opt, loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train,
                        y_train,
                        epochs=300,
                        verbose=0,
                        validation_data=(X_test, y_test),
                        callbacks=[callbacks])

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])

    from sklearn import metrics

    y_dnn_pred = model.predict_classes(X_test)

    y_dnn_pred = np.squeeze(y_dnn_pred)

    y_dnn_pred

    y_dnn_prob = model.predict(X_test)

    y_dnn_prob = np.squeeze(y_dnn_prob)

    y_dnn_prob

    print("Accuracy:", metrics.accuracy_score(y_test, y_dnn_pred))

    print(y_dnn_pred)

    print(y_dnn_prob)

    print(y_dnn_prob[y_dnn_prob > 0.35])

    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    """
    ergebnis = pd.DataFrame(columns=['Tag', 'Hagel', 'Vorhersage'])
    ergebnis['Tag'] = X_test.index
    ergebnis.set_index('Tag', inplace=True)
    ergebnis['Hagel'] = y_test
    ergebnis['Vorhersage'] = y_dnn_prob
    # print(y_test)
    # print(ergebnis.info)
    print('Tage an denen es gehagelt hat und die Vorhersage größer 0,5 war: ')
    print(ergebnis[(ergebnis['Vorhersage'] > 0.5) & (ergebnis['Hagel'] == 1.0)])
    print('Anzahl der Hageltage im Datensatz: ', ergebnis['Hagel'].sum())
    print('Hagelvorhersage > 0.5: ')
    print(ergebnis[ergebnis['Vorhersage'] > 0.5])
    """

    return
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import explained_variance_score, \
    mean_absolute_error, \
    median_absolute_error
from sklearn.model_selection import train_test_split
import tensorflow as tf


feldbergHageltage = pd.read_csv('produkt_wetter_tag_19490101_20140131_01346.txt', sep=";")

# Mess_Datum in Datentyp Datetime umwandeln und als Index setzen
feldbergHageltage['MESS_DATUM'] = pd.to_datetime(arg=feldbergHageltage['MESS_DATUM'], format='%Y%m%d')
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


# verschiedene Plots zum Wetter
"""
plt.subplot(1, 3, 1, title = 'Windstaerke Max')
#plt.plot(feldbergWetter.index, feldbergWetter['Schneehoehe'])
plt.plot(feldbergWetter.index, feldbergWetter['WindstaerkeMax'])
plt.subplot(1, 3, 2, title = 'Windstaerke Mittel')
plt.plot(feldbergWetter.index, feldbergWetter['WindstaerkeMittel'])
plt.subplot(1, 3, 3, title = 'Hagel')
plt.plot(feldbergWetter.index, feldbergWetter['Hagel'])
#plt.plot(feldbergWetter.index[1000:1100], feldbergWetter['WindstaerkeMax'][1000:1100])
plt.show()
"""

# verschiedene Boxplots um mögliche Zusammenhänge zu erkennen
"""
fig, ax = plt.subplots(1, 3)
sns.boxplot(x='Hagel', y='Niederschlagshoehe', data=feldbergWetter, ax=ax[0])
sns.boxplot(x='Hagel', y='Bewoelkung', data=feldbergWetter, ax=ax[1])
sns.boxplot(x='Hagel', y='RelativeFeuchteMittel', data=feldbergWetter, ax=ax[2])
plt.show()
"""

# Test ob ein Zusammenhang zwischen den Spalten zu erkennen ist
#print(feldbergWetter.corr())

# Droppen der Spalte Windstaerke Mittel, aufgrund zuvieler Fehlwerte
feldbergWetter = feldbergWetter.drop(columns=['WindstaerkeMittel'])

# Löschen aller Zeilen bis zum 1.1.1955
feldbergWetter.drop(feldbergWetter.loc['1949-01-01':'1954-12-31'].index, inplace=True)
# Fehlwerte ersetzen
feldbergWetter.replace(-999.0, np.nan, inplace=True)
# NaN-Werte werden durch linear interpolierte Werte, mit dem Wert vor und nach der Spalte ersetzt
feldbergWetter.interpolate(inplace=True)


# Plot WindstaerkeMax mit bereinigten Werten
"""
plt.plot(feldbergWetter.index, feldbergWetter['WindstaerkeMax'])
plt.xlabel('Datum')
plt.ylabel('Windstaerke')
plt.show()
"""

# Boxplots zum Vergleich der Spalten mit Hagel
"""
fig, ax = plt.subplots(2, 6)
sns.boxplot(x='Hagel', y='WindstaerkeMax', data=feldbergWetter, ax=ax[0, 0])
sns.boxplot(x='Hagel', y='Niederschlagshoehe', data=feldbergWetter, ax=ax[0, 1])
sns.boxplot(x='Hagel', y='Niederschlagsform', data=feldbergWetter, ax=ax[0, 2])
sns.boxplot(x='Hagel', y='Sonnenscheindauer', data=feldbergWetter, ax=ax[0, 3])
sns.boxplot(x='Hagel', y='Schneehoehe', data=feldbergWetter, ax=ax[0, 4])
sns.boxplot(x='Hagel', y='Bewoelkung', data=feldbergWetter, ax=ax[0, 5])
sns.boxplot(x='Hagel', y='DampfdruckMittel', data=feldbergWetter, ax=ax[1, 0])
sns.boxplot(x='Hagel', y='LuftdruckMittel', data=feldbergWetter, ax=ax[1, 1])
sns.boxplot(x='Hagel', y='TemperaturMittel', data=feldbergWetter, ax=ax[1, 2])
sns.boxplot(x='Hagel', y='RelativeFeuchteMittel', data=feldbergWetter, ax=ax[1, 3])
sns.boxplot(x='Hagel', y='LufttemperaturMax', data=feldbergWetter, ax=ax[1, 4])
sns.boxplot(x='Hagel', y='LufttemperaturMin2m', data=feldbergWetter, ax=ax[1, 5])
plt.show()
"""

# X will be a pandas dataframe of all columns except Hagel
X = feldbergWetter[[col for col in feldbergWetter.columns if col != 'Hagel']]

# y will be a pandas series of the meantempm
y = feldbergWetter['Hagel']

print(X.head(5))
# split data into training set and a temporary set using sklearn.model_selection.traing_test_split
#X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=23)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, shuffle=False)

# take the remaining 20% of data in X_tmp, y_tmp and split them evenly
#X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=23)
X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, shuffle=False)

X_train.shape, X_test.shape, X_val.shape
print("Training instances   {}, Training features   {}".format(X_train.shape[0], X_train.shape[1]))
print("Validation instances {}, Validation features {}".format(X_val.shape[0], X_val.shape[1]))
print("Testing instances    {}, Testing features    {}".format(X_test.shape[0], X_test.shape[1]))

feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]

regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                      hidden_units=[50, 50],
                                      model_dir='tf_wx_model')

def wx_input_fn(X, y, num_epochs=None, shuffle=True, batch_size=400):
    return tf.compat.v1.estimator.inputs.pandas_input_fn(x=X,
                                               y=y,
                                               num_epochs=num_epochs,
                                               shuffle=shuffle,
                                               batch_size=batch_size)




#regressor.train(input_fn=wx_input_fn(X_train,y_train, num_epochs=None, shuffle=False), steps=400)
"""
evaluations = []
STEPS = 400
for i in range(100):
    regressor.train(input_fn=wx_input_fn(X_train, y_train), steps=STEPS)
    evaluations.append(regressor.evaluate(input_fn=wx_input_fn(X_val,
                                                               y_val,
                                                               num_epochs=1,
                                                               shuffle=False)))
   """
"""
regressor.evaluate(input_fn=wx_input_fn(X_val, num_epochs=1, shuffle=False), steps=1)

predictions = regressor.predict(input_fn=wx_input_fn(X_test, num_epochs=1, shuffle=False), steps=1)
"""

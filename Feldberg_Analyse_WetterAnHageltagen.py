import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


feldbergHageltage = pd.read_csv('produkt_wetter_tag_19490101_20140131_01346.txt', sep=";")

# Mess_Datum in Datentyp Datetime umwandeln und als Index setzen
feldbergHageltage['MESS_DATUM'] = pd.to_datetime(arg=feldbergHageltage['MESS_DATUM'], format='%Y%m%d')
feldbergHageltage.set_index('MESS_DATUM', inplace=True)

feldbergWetter = pd.read_csv('produkt_klima_tag_19450101_20201231_01346.txt', sep=";")
feldbergWetter['MESS_DATUM'] = pd.to_datetime(arg=feldbergWetter['MESS_DATUM'], format='%Y%m%d')
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

# Anzeigen der Spalten
# print(feldbergWetter.columns)


pd.set_option('max_columns', None)
#print(feldbergWetter.head(20))

#print('Variable ist:'+str(feldbergWetter.iloc[[1],[1]]))

# -999 Werte in jeder Spalte zählen

ausgabe = {


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

data = pd.DataFrame(ausgabe)
print(data.sum())

"""
fehlwerteWindstaerkeMax = feldbergWetter.WindstaerkeMax==-999.0
fehlwerteWindstaerkeMittel = feldbergWetter.WindstaerkeMittel==-999.0
fehlwerteNiederschlagshoehe = feldbergWetter.Niederschlagshoehe==-999.0
fehlwerteNiederschlagsform = feldbergWetter.Niederschlagsform==-999.0
fehlwerteSonnenscheindauer = feldbergWetter.Sonnenscheindauer==-999.0
fehlwerteSchneehoehe = feldbergWetter.Schneehoehe==-999.0
fehlwerteBewoelkung = feldbergWetter.Bewoelkung==-999.0
fehlwerteDampfdruckMittel = feldbergWetter.DampfdruckMittel==-999.0
fehlwerteLuftdruckMittel = feldbergWetter.LuftdruckMittel==-999.0
fehlwerteTemperaturMittel = feldbergWetter.TemperaturMittel==-999.0
fehlwerteRelativeFeuchteMittel = feldbergWetter.RelativeFeuchteMittel==-999.0
fehlwerteLufttemperaturMax = feldbergWetter.LufttemperaturMax==-999.0
fehlwerteLufttemperaturMin2m = feldbergWetter.LufttemperaturMin2m==-999.0
fehlwerteLufttemperaturMin5cm = feldbergWetter.LufttemperaturMin5cm==-999.0
fehlwerteHagel = feldbergWetter.Hagel==-999.0"""
"""
data = pd.DataFrame([fehlwerteWindstaerkeMax, fehlwerteWindstaerkeMittel, fehlwerteNiederschlagshoehe, fehlwerteNiederschlagsform,
        fehlwerteSonnenscheindauer, fehlwerteSchneehoehe, fehlwerteBewoelkung, fehlwerteDampfdruckMittel,
        fehlwerteLuftdruckMittel, fehlwerteTemperaturMittel, fehlwerteRelativeFeuchteMittel, fehlwerteLufttemperaturMax,
        fehlwerteLufttemperaturMin2m, fehlwerteLufttemperaturMin5cm, fehlwerteHagel])
"""

#print(data.head(2))
#df_Fehlwerte = pd.DataFrame(data)





"""
feldbergNummer = feldbergWetter.apply(lambda x:True if -999.0 in list(x) else False, axis=1)
numOfRows = len(feldbergNummer[feldbergNummer==True].index)
print(numOfRows)
"""


windNummer = feldbergWetter.WindstaerkeMax==-999.0
print(windNummer.sum())


plt.plot(feldbergWetter.index, feldbergWetter['WindstaerkeMax'])
#plt.plot(feldbergWetter.index, feldbergWetter['WindstaerkeMittel'])
#plt.plot(feldbergWetter.index[1000:1100], feldbergWetter['WindstaerkeMax'][1000:1100])
plt.show()


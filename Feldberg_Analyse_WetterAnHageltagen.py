import numpy as np
import pandas as pd


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





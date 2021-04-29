import numpy as np
import pandas as pd



feldbergWetter = pd.read_csv('produkt_wetter_tag_19490101_20140131_01346.txt', sep=";")

#print(feldbergWetter[feldbergWetter['HAGEL']==1, feldbergWetter['HAGEL']].count())
print(feldbergWetter['HAGEL'].sum())
print('Summierte Hageltage am Feldberg seit 01.01.1949: ' + str(feldbergWetter['HAGEL'].sum()))
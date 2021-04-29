#Project Hail
#Last Worked on by: Chris
#Last Update on: 09/04/21
#To-Do: 
#Parameter wissenschaftlich bestimmen
#Live-Update Datenbank Zugang beschaffen 
#Front-End Display 


import numpy as np
import pandas as pd

TestWetter = pd.read_csv("TestWetter.csv")
print(TestWetter)

TWClean = TestWetter.drop(columns=['tavg','tmax', 'pres', 'wpgt', 'tsun', 'wspd', 'wdir' ])
TWClean['snow'] = TWClean['snow'].replace(np.nan, 0)
df = TWClean[TWClean.snow !=0]

Hail = df[TWClean.prcp >10]
Hail

print('On these days you should protect your PV-panels from Damage due to Hail:', Hail['date'])

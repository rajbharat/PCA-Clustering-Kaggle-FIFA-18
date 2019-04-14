# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:39:29 2019

@author: User
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing

fifa = pd.read_csv("C:/Users/User/Desktop/JMP/FIFA Cleansed Yes No.csv")

# Create LabelBinzarizer object
one_hot = LabelBinarizer()

# One-hot encode data

#fifa['Team'] = pd.Categorical(fifa['Team'])
#dfDummies = pd.get_dummies(fifa['Team'], prefix = 'country')
#print(dfDummies)
fifa['Ball Possession %'] = fifa['Ball Possession %'] / 100
fifa['Pass Accuracy %'] = fifa['Pass Accuracy %'] / 100

# Create x, where x the 'scores' column's values as floats
x = fifa[['Goal Scored','Ball Possession %','Attempts','On-Target','Off-Target','Blocked','Corners','Offsides','Free Kicks','Saves','Pass Accuracy %','Passes','Distance Covered (Kms)','Fouls Committed','1st Goal']].values.astype(float)

# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(x)
#print(type(x_scaled))
columns = ["Goal Scored","Ball Possession %","Attempts","On-Target","Off-Target","Blocked","Corners","Offsides","Free Kicks","Saves","Pass Accuracy %","Passes","Distance Covered (Kms)","Fouls Committed","1st Goal"]
# Run the normalizer on the dataframe
df_normalized = pd.DataFrame(x_scaled, columns=columns)
print(df_normalized.head())
fifa[['Goal Scored','Ball Possession %','Attempts','On-Target','Off-Target','Blocked','Corners','Offsides','Free Kicks','Saves','Pass Accuracy %','Passes','Distance Covered (Kms)','Fouls Committed','1st Goal']] = df_normalized[['Goal Scored','Ball Possession %','Attempts','On-Target','Off-Target','Blocked','Corners','Offsides','Free Kicks','Saves','Pass Accuracy %','Passes','Distance Covered (Kms)','Fouls Committed','1st Goal']]
print(fifa.head())
fifa.to_csv("C:/Users/User/Desktop/JMP/FIFA Normalized.csv")
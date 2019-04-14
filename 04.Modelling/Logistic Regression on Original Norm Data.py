# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 23:07:44 2019

@author: User
"""

import pandas as pd
# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('C:\\Users\\User\\Desktop\\JMP\\FIFA Normalized.csv')
feature_cols = ['GoalScored', 'BallPossession', 'Attempts', 'OnTarget','OffTarget','Blocked','Corners','Offsides','FreeKicks','Saves','PassAccuracy','Passes','DistanceCovered','FoulsCommitted'
                ,'YellowCard','YellowandRed','Red','1stGoal','PSO','GoalsinPSO','Owngoals','OwngoalTime']
data_filter=data[feature_cols]
target=data['ManoftheMatch']
target=target.values
print(data_filter)
print(target)
x_train, x_test, y_train, y_test = train_test_split(data_filter, target, test_size=0.20, random_state=0)
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
y_pred = logisticRegr.predict(x_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("R2 Score",r2_score(y_test,y_pred))
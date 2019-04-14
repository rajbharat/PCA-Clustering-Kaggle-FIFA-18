# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:20:57 2019

@author: User
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import csv

# Read dataset
data = pd.read_csv('C:\\Users\\User\\Desktop\\JMP\\FIFA Cluster.csv')

# Drop first column (not required)
feature_cols = ['Prin1', 'Prin2', 'Prin3', 'Prin4','Prin5','Prin6','Prin7']
data_filter=data[feature_cols]

# Normalize data (min/max scaling)
#data_arr = data_filter.values
#sc = preprocessing.MinMaxScaler()
#data_sc = sc.fit_transform(data_arr)
#data_filter = pd.DataFrame(data_sc)

SAMPLE_SIZE = 0.1
RANDOM_STATE = 42
NUM_CLUSTERS = 4     # k
NUM_ITER = 3          # n
NUM_ATTEMPTS = 20      # m

data_sample = data_filter.sample(frac=SAMPLE_SIZE, random_state=RANDOM_STATE, replace=False)
data_sample.shape

from sklearn.cluster import KMeans

km = KMeans(n_clusters=NUM_CLUSTERS, init='random', max_iter=1, n_init=1)#, verbose=1)
km.fit(data_sample)

#print('Pre-clustering metrics')
#print('----------------------')
#print('Inertia:', km.inertia_)
#print('Centroids:', km.cluster_centers_)

final_cents = []
final_inert = []
    
for sample in range(NUM_ATTEMPTS):
#    print('\nCentroid attempt: ', sample)
    km = KMeans(n_clusters=NUM_CLUSTERS, init='random', max_iter=1, n_init=1)#, verbose=1) 
    km.fit(data_sample)
    inertia_start = km.inertia_
    intertia_end = 0
    cents = km.cluster_centers_
        
    for iter in range(NUM_ITER):
        km = KMeans(n_clusters=NUM_CLUSTERS, init=cents, max_iter=1, n_init=1)
        km.fit(data_sample)
#        print('Iteration: ', iter)
#        print('Inertia:', km.inertia_)
#        print('Centroids:', km.cluster_centers_)
        inertia_end = km.inertia_
        cents = km.cluster_centers_

    final_cents.append(cents)
    final_inert.append(inertia_end)
    #print('Difference between initial and final inertia: ', inertia_start-inertia_end)
    
# Get best centroids to use for full clustering
best_cents = final_cents[final_inert.index(min(final_inert))]
best_cents

km_full = KMeans(n_clusters=NUM_CLUSTERS, init=best_cents, max_iter=100, verbose=1, n_init=1)
km_full.fit(data_filter)


labels = km_full.predict(data_filter)

data_labels=pd.DataFrame(labels, columns=['labels']) 

final_data=pd.concat([data_filter,data_labels],axis=1)

final_data.to_csv("C:\\Users\\User\\Desktop\\JMP\\FIFA Cluster_0704.csv")

data_test = pd.read_csv('C:\\Users\\User\\Desktop\\JMP\\FIFA Cluster_0704.csv')

data_test_slice=data_test[['Prin1', 'Prin2', 'Prin3', 'Prin4','Prin5','Prin6','Prin7']]

data_test_slice=data_test_slice[:30]

labels_predict = km_full.predict(data_test_slice)

test = pd.read_csv('C:\\Users\\User\\Desktop\\JMP\\FIFA Cluster_0704.csv')
test=test[:30]
test=test['labels']
test=test.values

print(type(test))
print(test)
cm = metrics.confusion_matrix(test, labels_predict)
print(cm)
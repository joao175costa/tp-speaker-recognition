#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 15:12:14 2018

@author: togepi
"""

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.mixture.gaussian_mixture import GaussianMixture

with open('full_pickle_features/Rob_Reiner.pickle', 'rb') as pickled:
    X_rob = pickle.load(pickled)

with open('full_pickle_features/Bob_Barker.pickle', 'rb') as pickled:
    X_bob = pickle.load(pickled)

y_rob = np.zeros_like(X_rob[:,0])
y_bob = np.ones_like(X_bob[:,0])

X = np.row_stack((X_rob, X_bob))
y = np.concatenate((y_rob,y_bob))

X_train, X_test, y_train, y_test = train_test_split(X,y)

#%%
gmm = GaussianMixture(n_components = 2)

gmm.fit(X_train,y_train)

#%%
y_pred = gmm.predict(X_test)

#%%
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
print(acc)
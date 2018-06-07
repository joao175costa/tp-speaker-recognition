#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 01:55:24 2018
@author: togepi
"""

import VoxModule as vox
import pickle

#with open(vox.GIT_PATH + 'loader_31_05.pkl', 'rb') as f:
#    feat_loader = pickle.load(f)

feat_loader = vox.VoxDataLoader()

#%%
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix

params = ParameterGrid({'n_components':[32,64,128,256],
          'n_iter':[5,10,20]})

rocs = {}
crs = {}
cms = {}

Xtrain = feat_loader.train_features
Xtest = feat_loader.test_features
ground_truth = vox.true_labels(Xtest)

imposter = feat_loader.imposter_features

for p in list(params):
    print(p)
    model = vox.GMM_UBM(**p)
    model.fit(Xtrain)
    pred = model.predict(Xtest)
    dec = model.dec_function(Xtest)
    key = (p['n_components'], p['n_iter'])
    crs[key] = classification_report(ground_truth, pred)
    cms[key] = confusion_matrix(ground_truth, pred)
    rocs[key] = vox.roc(ground_truth, dec)
    
#%%
import VoxModule as vox
import pickle

with open(vox.GIT_PATH + 'results_6_6.pkl','rb') as f:
    (cms, crs, rocs) = pickle.load(f)

list_it = [(256,5),(256,10),(256,20)]
list_c = [(32,10),(64,10),(128,10),(256,10)]
for r in list_c:
<<<<<<< Updated upstream
    vox.plot_det(rocs[r], r)
=======
vox.plot_det(rocs[r], r)
>>>>>>> Stashed changes

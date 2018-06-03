#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 00:11:31 2018

@author: togepi
"""

import VoxModule as vox

loader = vox.VoxDataLoader(n_speakers = 20, min_duration = 10)
Xtrain = loader.train_features

gmm_ubm = vox.GMM_UBM()
gmm_ubm.fit(Xtrain)

#%%
Xtest = loader.test_features
dec = gmm_ubm.dec_function(Xtest)
pred = gmm_ubm.predict(Xtest)
ground_truth = vox.true_labels(Xtest)

#%%
from sklearn.metrics import confusion_matrix, classification_report
cr = classification_report(ground_truth,pred)
cm = confusion_matrix(ground_truth,pred)
roc_curve = vox.roc(ground_truth, dec)
vox.plot_roc(roc_curve)

##%%
#from sklearn.svm import SVC
#svm_clf = SVC(probability = True)
#Xtrain_flat, ytrain_flat = loader.get_train_data()
#svm_clf.fit(Xtrain_flat, ytrain_flat)
##%%
#svm_probs = vox.predict_log_proba(svm_clf, Xtest)
##%%
#roc_svm = vox.roc(ground_truth, svm_probs)
#vox.plot_roc(roc_svm)
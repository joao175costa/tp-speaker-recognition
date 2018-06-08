#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 01:55:24 2018
@author: togepi
"""

import VoxModule as vox
import pickle

with open(vox.GIT_PATH + 'loader_31_05.pkl', 'rb') as f:
    feat_loader = pickle.load(f)

#feat_loader = vox.VoxDataLoader()

#%%
#from sklearn.model_selection import ParameterGrid
#from sklearn.metrics import classification_report, confusion_matrix
#
#params = ParameterGrid({'n_components':[32,64,128,256],
#          'n_iter':[5,10,20]})
#
#rocs = {}
#crs = {}
#cms = {}
#
#Xtrain = feat_loader.train_features
#Xtest = feat_loader.test_features
#ground_truth = vox.true_labels(Xtest)
#
#imposter = feat_loader.imposter_features
#
#for p in list(params):
#    print(p)
#    model = vox.GMM_UBM(**p)
#    model.fit(Xtrain)
#    pred = model.predict(Xtest)
#    dec = model.dec_function(Xtest)
#    key = (p['n_components'], p['n_iter'])
#    crs[key] = classification_report(ground_truth, pred)
#    cms[key] = confusion_matrix(ground_truth, pred)
#    rocs[key] = vox.roc(ground_truth, dec)
#    
#%%
import VoxModule as vox
import pickle
import numpy as np

with open(vox.GIT_PATH + 'results_6_6.pkl','rb') as f:
    (cms, crs, rocs) = pickle.load(f)

eers={}
cdets={}

for r in rocs:
    roc_data = rocs[r]
    fpr = roc_data[0]['macro']*100
    fnr = 100- 100*roc_data[1]['macro']
    diff = fpr-fnr
    diff_sign = np.sign(diff)
    i1 = np.where(diff_sign==-1)[0][-1]
    i2 = np.where(diff_sign==1)[0][0]
    x1 = fpr[i1] 
    x2 = fpr[i2]
    y1 = fnr[i1]
    y2 = fnr[i2]
    m = (y2-y1)/(x2-x1)
    b = y1-m*x1
    EER = b/(1-m)
    eers[r] = EER
    
    Pmiss = fnr/100
    Pfa = fpr/100
    Ptar = 0.1
    Cmiss = 1
    Cfa = 1
    Cdet = Cmiss*Pmiss*Ptar+Cfa*Pfa*(1-Ptar)
    Cdet_min = min(Cdet)
    cdets[r] = Cdet_min
    
for r in cms:
    cm = cms[r]
    acc = 100*np.sum(np.diag(cm))/np.sum(cm)
    print(r,acc)
print()
for r in eers:
    print(r,eers[r])
    
#%%

list_it = [(128,5),(128,10),(128,20)]
list_c = [(32,10),(64,10),(128,10),(256,10)]
for r in list_it:
    vox.plot_det(rocs[r], eers[r], r)
    
#%%
import pickle
import VoxModule as vox

with open(vox.GIT_PATH + 'results_8_6.pkl','rb') as f:
    (cms, crs, accs,eers,rocs) = pickle.load(f)

#%%
import matplotlib
import matplotlib.pyplot as plt

list_det = [(256,10),(64,10),(16,10),(4,10),(1,10)]
user = 'macro'
fig, ax= plt.subplots()

for r in list_det:
    roc_data = rocs[r]
    fpr = roc_data[0]
    tpr = roc_data[1]
    eer_data = eers[r]

    lw = 2
    plt.plot(fpr[user]*100, 100-tpr[user]*100,
             lw=lw, 
             label='C='+str(r[0]))
    plt.plot(eer_data, eer_data, 'rx')
    
ax.loglog()

major = matplotlib.ticker.MultipleLocator(5)
ax.xaxis.set_major_locator(major)
ax.yaxis.set_major_locator(major)
ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

#ax.set_xlim([10, 60])
#ax.set_ylim([10, 60])
plt.title('DET Curves')
plt.xlabel('False Positive Rate (%)')
plt.ylabel('False Negative Rate (%)')
ax.grid(True, linestyle='--')
ax.legend()
plt.show()

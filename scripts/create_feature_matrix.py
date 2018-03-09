# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import pickle
import numpy as np

pickleFiles = os.listdir('pickle_features/')


#%%
for file in pickleFiles:
    print(file)
    if (not os.path.isfile('full_pickle_features/'+file)):
        features = []
        with open('pickle_features/'+file,'rb') as f:
            list_feat = pickle.load(f)
        for i in range(np.shape(list_feat)[0]):
            print(i, end = ' ')
            for j in range(np.shape(list_feat[i])[0]):
                features.append(list_feat[i][j])
        features = np.array(features)
        with open('full_pickle_features/'+file, 'wb') as out:
            pickle.dump(features, out, pickle.HIGHEST_PROTOCOL)
        print('features saved')
    
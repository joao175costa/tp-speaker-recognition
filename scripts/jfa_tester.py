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
with open(vox.GIT_PATH + 'gmmubm_31_05.pkl', 'rb') as f:
    gmm_ubm_model = pickle.load(f)
    
iv = vox.iVectors()
iv.load_GMM_UBM(gmm_ubm_model)
iv.train_T(feat_loader.train_features)

T = iv.T_matrix

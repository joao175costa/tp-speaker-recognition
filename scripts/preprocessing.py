#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 22:43:37 2018

@author: togepi
"""

import numpy as np
import os
import pickle
from speechpy.processing import cmvn


list_speakers = os.listdir('pickle_features/')
for speaker in list_speakers:
    print(speaker)
    with open('pickle_features/'+speaker,'rb') as pickled:
        feats = pickle.load(pickled)
    processed_feats = []
    for frame in feats:
        processed_frame = cmvn(frame, variance_normalization=True)
        processed_feats.append(processed_frame)
    with open('processed_pickle_features/'+speaker,'wb') as out:
        pickle.dump(processed_feats, out, pickle.HIGHEST_PROTOCOL)
        

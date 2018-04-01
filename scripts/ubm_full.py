#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 22:31:47 2018

@author: togepi
"""
import numpy as np
import os
import pickle as pkl

GIT_PATH = '/home/togepi/feup-projects/tp-speaker-recognition/'
HDD_PATH = '/run/media/togepi/USB HDD/FEUP/VoxCeleb/'
FEATURES_PATH = '/run/media/togepi/USB HDD/FEUP/VoxCeleb/pickle_features_new/'

def get_total_shape():
    total_samples = 0
    speaker_list = os.listdir(FEATURES_PATH)
    for speaker in speaker_list:
        with open(FEATURES_PATH + speaker,'rb') as file:
            speaker_samples = pkl.load(file)
        for utterance in speaker_samples:
            total_samples += np.shape(utterance)[0]
    total_features = np.shape(utterance)[1]
    return (total_samples, total_features)

#%% 
def set_features(memmap):
    index = 0
    i = 0
    speaker_list = os.listdir(FEATURES_PATH)
    total_speakers = len(speaker_list)
    for speaker in speaker_list:
        with open(FEATURES_PATH + speaker,'rb') as file:
            speaker_samples = pkl.load(file)
        for utterance in speaker_samples:
            utt_length = np.shape(utterance)[0]
            memmap[index:index+utt_length,:] = utterance
            index += utt_length
        i+=1
        print(i,'/',total_speakers)
 
#%%       
#total_shape = get_total_shape()
# (126421548,39)

#data_ubm = np.memmap(GIT_PATH + 'data.dat', dtype='float32', mode = 'w+', shape = total_shape)

#%%
#set_features(data_ubm)

#%%

data_ubm = np.memmap(GIT_PATH+'data.dat', dtype='float32', mode='r', shape = (126421548,39))

from sklearn.mixture import GaussianMixture
UBM = GaussianMixture(n_components=1024, max_iter = 10, init_params = 'kmeans', verbose = 10, verbose_interval = 1)
UBM.fit(data_ubm)

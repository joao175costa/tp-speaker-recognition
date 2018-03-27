#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:26:18 2018

@author: togepi
"""

from scipy.io.wavfile import read as read_wav
from speechpy.feature import mfcc, extract_derivative_feature
from speechpy.processing import preemphasis, cmvn
import os
import numpy as np
import pickle

HDD_PATH = '/run/media/togepi/USB HDD/FEUP/VoxCeleb'
FEATURES_PATH = '/run/media/togepi/USB HDD/FEUP/VoxCeleb/pickle_features_new/'

def pre_processing(wav_signal):
    processed = preemphasis(wav_signal)
    return processed
    
def get_features(speaker, file):
    # read the .wav file and obtain the signal and Fs from it
    filepath = 'voxceleb1_wav/'+speaker+'/'+file
    wav = read_wav(filepath)
    signal = wav[1]
    Fs = wav[0]
    
    # preenphasis of higher frequencies
    signal = pre_processing(signal)
    
    # mfcc extraction
    feats_mfcc = mfcc(signal,Fs)
    
    #D and DD MFCC calculation
    feats_delta = extract_derivative_feature(feats_mfcc)
    
    # reshape array so it is 2D, with MFCC, DMFCC and DDMFCC
    feats_shape = np.shape(feats_delta)
    feats_delta = np.reshape(feats_delta,(feats_shape[0],feats_shape[1]*feats_shape[2]))
    
    # post processing
    feats_post = cmvn(feats_delta, variance_normalization = True)
    return feats_post

def get_features_from_speaker(speaker):
    path = 'voxceleb1_wav/'+speaker
    files = os.listdir(path)
    total_files = len(files)
    i=0
    features_speaker = []
    for file in files:
        features = get_features(speaker,file)
        features_speaker.append(features)
        i+=1
        print(i,'/',total_files)
    with open(FEATURES_PATH +speaker+'.pkl', 'wb') as f:
        pickle.dump(features_speaker, f, pickle.HIGHEST_PROTOCOL)

def get_features_all_speakers():
    os.chdir(HDD_PATH)
    speakers = os.listdir('voxceleb1_wav/')
    total_speakers = len(speakers)
    i=0
    for speaker in speakers:
        get_features_from_speaker(speaker)
        i+=1
        print(i,'/',total_speakers)
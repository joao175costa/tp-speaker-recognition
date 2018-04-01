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
WAV_PATH = '/run/media/togepi/USB HDD/FEUP/VoxCeleb/voxceleb1_wav/'
FEATURES_PATH = '/run/media/togepi/USB HDD/FEUP/VoxCeleb/pickle_features_new/'

def calculate_record_duration_speaker(speaker):
    # calculates the number of minutes of recording for each speaker
    wav_files = os.listdir(WAV_PATH + speaker)
    total_seconds = 0
    for file in wav_files:
        filepath = WAV_PATH + speaker + '/' + file
        wav = read_wav(filepath)
        signal = wav[1]
        ts = 1/wav[0]
        signal_duration = len(signal) * ts
        total_seconds += signal_duration
    total_minutes = total_seconds/60
    return total_minutes
    
def calculate_record_duration_all():
    speakers = os.listdir(WAV_PATH)
    list_of_duration = []
    i=0
    total_speakers = len(speakers)
    for speaker in speakers:
        i+=1
        print(i,'/',total_speakers)
        duration =  calculate_record_duration_speaker(speaker)
        list_of_duration.append((speaker, duration))
    list_of_duration.sort(key = lambda speaker: speaker[1], reverse=True)
    return list_of_duration

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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:26:18 2018

@author: togepi
"""

from scipy.io.wavfile import read as read_wav
from speechpy.feature import mfcc
import os
import pickle
    
def get_mfcc(speaker, file):
    filepath = 'voxceleb1_wav/'+speaker+'/'+file
    wav = read_wav(filepath)
    signal = wav[1]
    Fs = wav[0]
    feat = mfcc(signal,Fs)
    return feat

def get_mfcc_from_speaker(speaker):
    path = 'voxceleb1_wav/'+speaker
    files = os.listdir(path)
    total_files = len(files)
    i=0
    mfcc_speaker = []
    for file in files:
        features = get_mfcc(speaker,file)
        mfcc_speaker.append(features)
        i+=1
        print(i,'/',total_files)
    with open(speaker+'.pickle', 'wb') as f:
        pickle.dump(mfcc_speaker, f, pickle.HIGHEST_PROTOCOL)

def get_mfcc_all():
    speakers = os.listdir('voxceleb1_wav/')
    total_speakers = len(speakers)
    i=0
    for speaker in speakers:
        get_mfcc_from_speaker(speaker)
        i+=1
        print(i,'/',total_speakers)
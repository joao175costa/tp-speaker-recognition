#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 15:05:09 2018

@author: togepi
"""
import numpy as np
import os
import pickle as pkl
import itertools
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, classification_report

GIT_PATH = '/home/togepi/feup-projects/tp-speaker-recognition/'
HDD_PATH = '/run/media/togepi/USB HDD/FEUP/VoxCeleb/'
FEATURES_PATH = '/run/media/togepi/USB HDD/FEUP/VoxCeleb/pickle_features_new/'

def load_user_data(filename):
    with open(FEATURES_PATH+filename,'rb') as f:
        data = pkl.load(f)
    return data

def load_duration():
    with open(GIT_PATH + 'durations.pkl', 'rb') as f:
        duration = pkl.load(f)
    return duration

def get_features_from_files(filelist):
    f_matrix = []
    train_data = []
    test_data = []
    for file in filelist:
        userData = load_user_data(file)
        userData_train, userData_test = train_test_split(userData)
        train_data.append((userData_train,file))
        test_data.append((userData_test,file))
    return train_data, test_data

def flatten_speaker_feature_matrix(speakerData):
    f_matrix=[]
    data = speakerData[0]
    label = speakerData[1]
    for utt in data:
            for sample in utt:        
                f_matrix.append(sample)
    f_matrix = np.array(f_matrix)
    return f_matrix, label
                    
def flatten_full_feature_matrix(featureData):
    f_matrix = []
    for speaker in featureData:
        arr,_= flatten_speaker_feature_matrix(speaker)
        f_matrix.append(arr)
    f_matrix=np.concatenate(f_matrix)
    return f_matrix

def get_speakers(n_speakers = 5, min_duration = 5):
    # create the ubm model with n_speakers
    # there are too many speakers to compute the full UBM, so a small population
    # is selected. if random=True, this population is random (may lead to bias)
    # Otherwise, the n speakers with more samples are selected
    
    speaker_duration = load_duration()
    
    min_speakers = [list(group) for val, group in itertools.groupby(speaker_duration,
         lambda x: x[1] > min_duration) if val]

    selected_speakers = min_speakers[0]
    selected_speakers = selected_speakers[-n_speakers:]
    
    # create the feature matrix for training
    selected_files = [auth + '.pkl' 
                      for auth in np.array(selected_speakers)[:,0]]
    X_train, X_test = get_features_from_files(selected_files)
    X_train_flat = flatten_full_feature_matrix(X_train)
    return X_train_flat, X_train, X_test

def train_gmm(feat_matrix, n_components = 1024, save=False):
    gmm = BayesianGaussianMixture(n_components = n_components, 
                                  max_iter = 50,
                                  covariance_type = 'diag',
                                  verbose=10,
                                  verbose_interval=1)
    gmm.fit(feat_matrix)
    if save:
        with open(GIT_PATH + 'UBM.pkl','wb') as f:
            pkl.dump(gmm, f, -1)
            print('Saved')
    return gmm

def get_adapted_speaker_models(UBMModel, speaker_training_data, 
                               Rfactors = [16,16,16],
                               save = False):
   
    UBMcomponents = UBMModel.get_params()['n_components']
    UBMcov_type = UBMModel.get_params()['covariance_type']
    UBMweights = UBMModel.weights_
    UBMmeans = UBMModel.means_
    UBMcov = UBMModel.covariances_
    
    GMM_list_speakers = []
    
    size = len(speaker_training_data)
    i=0
    for speakerData in speaker_training_data:
        i+=1
        print(i, size)
        speaker_features, name = flatten_speaker_feature_matrix(speakerData)
        total_samples = np.shape(speaker_features)[0]
        n_features = np.shape(speaker_features)[1]
        probs_training_UBM = UBMModel.predict_proba(speaker_features)
        
        Pr_comp_sample = (probs_training_UBM * UBMweights)
        Pr_sum = np.sum(Pr_comp_sample, axis=1)
        Pr_comp_sample  = Pr_comp_sample / Pr_sum[:,None]
        
        # sufficient parameters for estimation
        count_n_i = np.sum(Pr_comp_sample, axis=0)
        first_moment_i = np.dot(Pr_comp_sample.T, speaker_features)/count_n_i[:,None]
        second_moment_i = np.dot(Pr_comp_sample.T, speaker_features**2)/count_n_i[:,None]
        
        # adjust model parameters for each speaker:
        alpha_w = count_n_i/(count_n_i+Rfactors[0])
        alpha_m = count_n_i/(count_n_i+Rfactors[1])
        alpha_c = count_n_i/(count_n_i+Rfactors[2])
        
        adj_weights = alpha_w*count_n_i/total_samples + (1-alpha_w)*UBMweights
        adj_weights = normalize(adj_weights.reshape(1,-1), norm='l1')
        adj_means = alpha_m[:,None]*first_moment_i + (1-alpha_m)[:,None]*UBMmeans
        adj_cov = alpha_c[:,None]*second_moment_i + (1-alpha_c)[:,None]*(UBMcov+UBMmeans**2) - adj_means**2
        
        # created adapted GMM
        GMMspeaker = GaussianMixture(n_components = UBMcomponents,
                                     covariance_type = UBMcov_type)
        # for the model to work it needs to be fitted to random data first
        GMMspeaker.fit(np.random.rand(UBMcomponents,n_features))
        # then the parameters are set
        GMMspeaker.weights_ = adj_weights
        GMMspeaker.means_ = adj_means
        GMMspeaker.covariances_ = adj_cov
        GMM_list_speakers.append((name,GMMspeaker))
    if save:
        with open(GIT_PATH + 'adapted_GMM.pkl', 'wb') as f:
            pkl.dump(GMM_list_speakers, f, -1)
    
    return GMM_list_speakers

def score_utterance(utt, UBM, GMM_list, prob=False):
    # Score an utterance based on the sum of the log likelihood of its samples
    # Does this for the UBM and for each speaker adapted GMM
    
    log_likelihood_ubm = UBM.score(utt)
    log_likelihood_gmm = []
    for model in GMM_list:
        gmm = model[1]
        log_likelihood = gmm.score(utt)
        log_likelihood_gmm.append(log_likelihood)
    log_likelihood_gmm = np.array(log_likelihood_gmm)
    scores = log_likelihood_gmm - log_likelihood_ubm
    if prob:
        log_likelihood_gmm = np.array(log_likelihood_gmm)
        log_likelihood_ubm = [log_likelihood_ubm for gmm in log_likelihood_gmm]
        log_likelihood_ubm = np.array(log_likelihood_ubm)
        return np.stack((log_likelihood_gmm, log_likelihood_ubm)).T
    return scores

def predict_label(utt, UBM, GMM_list):
    # The 
    scores = score_utterance(utt, UBM, GMM_list)
    best_index = np.argmax(scores)
    best_label = GMM_list[best_index][0]
    return best_label

def predict_test_set(Xtest, UBM, GMM_list):
    predicted_labels = []
    true_labels = []

    for speaker in Xtest:
        label = speaker[1]
        list_utt = speaker[0]
        for utt in list_utt:
            predicted_label = predict_label(utt, UBM, GMM_list)
            predicted_labels.append(predicted_label)
            true_labels.append(label)
    return predicted_labels, true_labels
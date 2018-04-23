#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:43:18 2018

@author: togepi
"""
import itertools
import operator
import pickle as pkl
import numpy as np
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp

GIT_PATH = '/home/togepi/feup-projects/tp-speaker-recognition/'
HDD_PATH = '/run/media/togepi/USB HDD/FEUP/VoxCeleb/'
FEATURES_PATH = '/run/media/togepi/USB HDD/FEUP/VoxCeleb/pickle_features_new/'

def flatten_speaker_feature_matrix(speakerData):
    f_matrix=[]
    data = speakerData
    for utt in data:
            for sample in utt:
                f_matrix.append(sample)
    f_matrix = np.array(f_matrix)
    return f_matrix

def flatten_full_feature_matrix(featureData):
    f_matrix = []
    labels = []

    for speaker in featureData:
        arr = flatten_speaker_feature_matrix(featureData[speaker])
        f_matrix.append(arr)
        n_labels = len(arr)
        labels.append(np.full(n_labels,speaker))

    f_matrix=np.concatenate(f_matrix)
    labels = np.concatenate(labels)
    return f_matrix, labels

class VoxDataLoader:
    def __init__(self, n_speakers=5, min_duration=5):
        self.n_speakers = n_speakers
        self.min_duration = min_duration
        self.features = None
        self.train_features = None
        self.test_features = None

        self.load_features()
        self.split()

    def load_user_data(self,filename):
        with open(FEATURES_PATH+filename,'rb') as f:
            data = pkl.load(f)
        return data

    def load_duration(self):
        with open(GIT_PATH + 'durations.pkl', 'rb') as f:
            duration = pkl.load(f)
        return duration

    def load_features(self):
        # there are too many speakers to compute the full UBM, so a small population
        # is selected. if random=True, this population is random (may lead to bias)
        # Otherwise, the n speakers with more samples are selected
        self.features = {}

        speaker_duration = self.load_duration()

        min_speakers = [list(group) for val, group in itertools.groupby(speaker_duration,
                        lambda x: x[1] > self.min_duration) if val]

        selected_speakers = min_speakers[0]
        selected_speakers = selected_speakers[-self.n_speakers:]

        # create the feature matrix for training
        for auth in sorted(np.array(selected_speakers)[:,0]):
            auth_pkl = auth + '.pkl'
            self.features[auth] = self.load_user_data(auth_pkl)
            
    def split(self):
        train_data = {}
        test_data = {}
        for speaker in sorted(self.features):
            train_data[speaker], test_data[speaker] = train_test_split(self.features[speaker])
        self.train_features = train_data
        self.test_features = test_data

    def get_train_data(self):
        X_train, y_train = flatten_full_feature_matrix(self.train_features)
        return X_train, y_train

class GMM_UBM:
    def __init__(self, n_components=128, n_iter=10, r_factors = [16,16,16]):
        self.n_components = n_components
        self.n_iter = n_iter
        self.r_factors = r_factors
        # placeholders
        self.ubm = None
        self.adapted_gmm = None
        self.is_fitted = False

    def fit(self, X):
        # create the UBM from all the features
        self.ubm = BayesianGaussianMixture(n_components = self.n_components,
                                      max_iter = self.n_iter,
                                      covariance_type = 'diag',
                                      verbose=1000,
                                      verbose_interval=1)
        flatX, _ = flatten_full_feature_matrix(X)
        self.ubm.fit(flatX)
        
        # create adapted gmm from ubm
        print('Creating adapted GMMs from UBM')
        UBMcomponents = self.n_components
        UBMcov_type = 'diag'
        UBMweights = self.ubm.weights_
        UBMmeans = self.ubm.means_
        UBMcov = self.ubm.covariances_

        self.adapted_gmm = {}

        size = len(X)
        i=0
        for speaker in X:
            i+=1
            print(i,'/', size)
            speaker_features = flatten_speaker_feature_matrix(X[speaker])
            total_samples = np.shape(speaker_features)[0]
            n_features = np.shape(speaker_features)[1]
            probs_training_UBM = self.ubm.predict_proba(speaker_features)

            Pr_comp_sample = (probs_training_UBM * UBMweights)
            Pr_sum = np.sum(Pr_comp_sample, axis=1)
            Pr_comp_sample  = Pr_comp_sample / Pr_sum[:,None]

            # sufficient parameters for estimation
            count_n_i = np.sum(Pr_comp_sample, axis=0)
            first_moment_i = np.dot(Pr_comp_sample.T, speaker_features)/count_n_i[:,None]
            second_moment_i = np.dot(Pr_comp_sample.T, speaker_features**2)/count_n_i[:,None]

            # adjust model parameters for each speaker:
            alpha_w = count_n_i/(count_n_i+self.r_factors[0])
            alpha_m = count_n_i/(count_n_i+self.r_factors[1])
            alpha_c = count_n_i/(count_n_i+self.r_factors[2])

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
            self.adapted_gmm[speaker] = GMMspeaker

        self.is_fitted = True

    def predict_utt(self, utt):
        size = len(utt)
        log_likelihood_ubm = self.ubm.score_samples(utt).sum()/size
        log_likelihoods = {}
        for speaker in self.adapted_gmm:
            adapted_score = self.adapted_gmm[speaker].score_samples(utt).sum()/size
            log_likelihoods[speaker] = adapted_score - log_likelihood_ubm
        log_likelihoods['UBM'] = log_likelihood_ubm
        return log_likelihoods

    def predict(self, X):
        predictions = []
        for speaker in X:
            for utt in X[speaker]:
                scores = self.predict_utt(utt)
                del scores['UBM']
                best_label = max(scores.items(), key=operator.itemgetter(1))[0]
                predictions.append(best_label)
        return np.array(predictions)

    def dec_function(self, X):
        predictions = []
        for speaker in X:
            for utt in X[speaker]:
                scores = self.predict_utt(utt)
                del scores['UBM']
                scores_utt = []
                for classes in scores:
                    scores_utt.append(scores[classes])
                predictions.append(scores_utt)
        predictions = np.array(predictions)
        
        # convert from log space to probabilities
        predictions_sub = predictions - predictions.max(axis=1, keepdims=True)
        predictions_exp = np.exp(predictions_sub)
        predictions_norm = predictions_exp/predictions_exp.sum(axis=1, keepdims=True)
        return predictions_norm

def true_labels(X):
    true_labels = []
    for speaker in X:
        for utt in X[speaker]:
            true_labels.append(speaker)
    return np.array(true_labels)

def predict_log_proba(clf, Xt):
    total_utt = 0
    for speaker in Xt:
        total_utt+=len(Xt[speaker])
    n_classes = len(clf.predict_log_proba(Xt[speaker][0]).sum(axis=0, keepdims=True))
    log_probs = np.zeros((total_utt,n_classes))
    i=0
    for speaker in Xt:
        for utt in Xt[speaker]:
            log_probs[i]=clf.predict_log_proba(utt).sum(axis=0, keepdims = True)[0]
            i+=1
            print(i)
    log_probs = np.array(log_probs)
    predictions_sub = log_probs - log_probs.max(axis=1, keepdims=True)
    predictions_exp = np.exp(predictions_sub)
    predictions_norm = predictions_exp/predictions_exp.sum(axis=1, keepdims=True)
    return predictions_norm

def roc(ytrue, ypred):
    classes = np.unique(ytrue)
    n_classes = len(classes)
    ytest_bin = label_binarize(ytrue, classes)
    Xtest_dec = ypred
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(ytest_bin[:, i], Xtest_dec[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr["micro"], tpr["micro"], _ = roc_curve(ytest_bin.ravel(), Xtest_dec.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    return(fpr, tpr, roc_auc)
    
def plot_roc(roc_data, user = 'micro'):
    fpr = roc_data[0]
    tpr = roc_data[1]
    roc_auc = roc_data[2]
    
    plt.figure()
    lw = 2
    plt.plot(fpr[user], tpr[user], color='darkorange',
             lw=lw, label='ROC curve (area = %0.8f)' % roc_auc[user])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic ('+str(user)+')')
    plt.legend(loc="lower right")
    plt.show()
    

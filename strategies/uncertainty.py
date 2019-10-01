#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:45:41 2019

@author: dhruv.sharma
"""

"""
Uncertainty measures and uncertainty based sampling strategies for the active learning models.
"""
import numpy as np
from sklearn.exceptions import NotFittedError
from scipy.special import entr
from keras.models import Model
from tqdm import tqdm
from skimage.measure import block_reduce

def mc_dropout_uncertainty(model, X, nb_mc = 10):
    """
    MC Dropout uncertainty for the model as described in https://arxiv.org/pdf/1506.02142.pdf
    
    Args:
        
    """
    print("#####################################")
    print("###### MC Dropout Sampling ##########")
    print("#####################################")

    model = model.model
    model_MC = Model(inputs=model.input, outputs=model.layers[-1].output)
#    model_MC = K.function(
#            [model.layers[0].input, 
#             K.learning_phase()],
#            [model.layers[-1].output])
            
    result = []
    print("MC runs through the network")
    
    for _ in tqdm(range(nb_mc)):
#        out = np.array(model_MC([X , 1])[0])  
        out = model_MC.predict(X, batch_size = 4)
        result.append(out)
    
    MC_samples = np.array(result)
    
    expected_entropy = - np.mean(np.sum(MC_samples * np.log(MC_samples + 1e-10), axis=-1), axis=0)  # [batch size]
    expected_p = np.mean(MC_samples, axis=0)
    entropy_expected_p = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)  # [batch size]
    BALD_acq = entropy_expected_p - expected_entropy
    
    BALD_acq = BALD_acq.mean(axis = (1,2))
    
    del model_MC
    return BALD_acq

def spatial_uncertainty(model, X, threshold = 0.5, nb_mc = 10):
    """
    Spatial uncertainty taken into account to capture a better uncertainty result.
    MC Dropout uncertainty for the model as described in https://arxiv.org/pdf/1506.02142.pdf
    
    Args:
        
    """
    print("#####################################")
    print("###### MC Dropout Sampling ##########")
    print("#####################################")

    model = model.model
    model_MC = Model(inputs=model.input, outputs=model.layers[-1].output)
#    model_MC = K.function(
#            [model.layers[0].input, 
#             K.learning_phase()],
#            [model.layers[-1].output])
            
    result = []
    print("MC runs through the network")
    
    for _ in tqdm(range(nb_mc)):
#        out = np.array(model_MC([X , 1])[0])  
        out = model_MC.predict(X, batch_size = 4)
        result.append(out)
    
    MC_samples = np.array(result)
    
    expected_entropy = - np.mean(np.sum(MC_samples * np.log(MC_samples + 1e-10), axis=-1), axis=0)  # [batch size]
    expected_p = np.mean(MC_samples, axis=0)
    entropy_expected_p = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)  # [batch size]
    
    BALD_acq = entropy_expected_p - expected_entropy
    
#    ############################
#    from PIL import Image
#    data = -1*BALD_acq[0]*255.0
#    img = Image.fromarray(data)
#    img.show()
#    ############################
    
    
    pooledBALD = block_reduce(BALD_acq, block_size=(1,4,4), func=np.mean)
    
    elements = 1
    for dim in np.shape(pooledBALD)[1:]:
        elements *= dim
    
    uncertain_blocks_score = np.sum(pooledBALD>threshold, axis=(1,2))
    uncertain_blocks_score = uncertain_blocks_score/(1.0*elements)

    return uncertain_blocks_score

def segmentation_uncertainty(model, X):
    """
    Segmentation uncertainty of the model for the provided samples.

    Args:
        model: The model for which the uncertainty is to be measured.
        X: The samples for which the uncertainty of segmentation is to be measured.

    Returns:
        model uncertainty, which is 1 - P(prediction is correct).
    """
    # calculate uncertainty for each point provided
    try:
        segment_uncertainty = model.predict(X, verbose = 0)
    except NotFittedError:
        return np.ones(shape=(X.shape[0], ))
    
    print(segment_uncertainty[0][0])
    # for each point, select the maximum uncertainty
    uncertainty = 1 - np.mean(np.max(segment_uncertainty, axis=len(X.shape)-1), axis = (1,2))
    return uncertainty


def segmentation_margin(model, X):
    """
    Segmentation margin uncertainty of the model for the provided samples. This uncertainty measure takes the
    first and second most likely predictions and takes the difference of their probabilities, which is the margin.

    Args:
        model: The model for which the prediction margin is to be measured.
        X: The samples for which the prediction margin of segmentation is to be measured.

    Returns:
        Margin uncertainty, which is the difference of the probabilities of first and second most likely predictions.
    """
    try:
        segment_uncertainty = model.predict(X)
    except NotFittedError:
        return np.zeros(shape=(X.shape[0], ))

    if segment_uncertainty.shape[1] == 1:
        return np.zeros(shape=(segment_uncertainty.shape[0],))

    part = np.partition(-1*segment_uncertainty, 1, axis=len(X.shape)-1)
    margin = np.mean((- part[:, :, :, 0] + part[:, :, :, 1]), axis = (1,2))

    return margin


def segmentation_entropy(model, X):
    """
    Entropy of predictions of the for the provided samples.

    Args:
        model: The model for which the prediction entropy is to be measured.
        X: The samples for which the prediction entropy is to be measured.

    Returns:
        Entropy of the class probabilities.
    """
    try:
        segment_uncertainty = model.predict(X)
    except NotFittedError:
        return np.zeros(shape=(X.shape[0], ))

    entropy = np.mean(entr(segment_uncertainty).sum(axis= len(X.shape)-1), axis = (1,2))
    
    return entropy

def spatial_unceratinty_sampling(model, X_u, nb_mc = 10, n_instances = 1):
    """
    Spatial uncertainty sampling query strategy. Selects the least sure instances for labelling.

    Args:
        model: The model for which the labels are to be queried.
        X_u: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        nb_mc: Number of monte-carlo steps

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X_u chosen to be labelled.
    """
    uncertainty = spatial_uncertainty(model, X_u, nb_mc)
    query_idx = np.argpartition(-uncertainty, n_instances-1, axis=0)[:n_instances]
    
    return query_idx, X_u[query_idx]

def mc_dropout_sampling(model, X_u, nb_mc = 10, n_instances = 1):
    """
    MC Dropout based sampling query strategy. Selects the least sure instances for labelling.

    Args:
        model: The model for which the labels are to be queried.
        X_u: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        nb_mc: Number of monte-carlo steps

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X_u chosen to be labelled.
    """
    uncertainty = mc_dropout_uncertainty(model, X_u, nb_mc)
    query_idx = np.argpartition(-uncertainty, n_instances-1, axis=0)[:n_instances]
    
    return query_idx, X_u[query_idx]

def uncertainty_sampling(model, X_u, n_instances = 1):
    """
    Uncertainty sampling query strategy. Selects the least sure instances for labelling.

    Args:
        model: The model for which the labels are to be queried.
        X_u: The pool of samples to query from.
        n_instances: Number of samples to be queried.

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X_u chosen to be labelled.
    """
    uncertainty = segmentation_uncertainty(model, X_u)
    query_idx = np.argpartition(-uncertainty, n_instances-1, axis=0)[:n_instances]

    return query_idx, X_u[query_idx]


def margin_sampling(model, X_u, n_instances = 1):
    """
    Margin sampling query strategy. Selects the instances where the difference between
    the first most likely and second most likely classes are the smallest.
    Args:
        model: The model for which the labels are to be queried.
        X_u: The pool of samples to query from.
        n_instances: Number of samples to be queried.
        
    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X_u chosen to be labelled.
    """
    margin = segmentation_margin(model, X_u)
    query_idx = np.argpartition(-margin, n_instances-1, axis=0)[:n_instances]

    return query_idx, X_u[query_idx]


def entropy_sampling(model, X_u, n_instances = 1):
    """
    Entropy sampling query strategy. Selects the instances where the class probabilities
    have the largest entropy.

    Args:
        model: The model for which the labels are to be queried.
        X_u: The pool of samples to query from.
        n_instances: Number of samples to be queried

    Returns:
        The indices of the instances from X chosen to be labelled;
        the instances from X chosen to be labelled.
    """
    entropy = segmentation_entropy(model, X_u)
    query_idx = np.argpartition(-entropy, n_instances-1, axis=0)[:n_instances]

    return query_idx, X_u[query_idx]

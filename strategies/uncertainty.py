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

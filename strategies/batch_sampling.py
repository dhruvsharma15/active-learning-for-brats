#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:39:29 2019

@author: dhruv.sharma
"""

"""
Uncertainty measures that explicitly support batch-mode sampling for active learning models.
"""

from typing import Callable, Optional, Tuple, Union
from tqdm import tqdm

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin_min
from keras.models import Model
from strategies.uncertainty import *


def select_instance(
        X_training,
        X_pool,
        X_training_feat,
        X_pool_feat,
        X_uncertainty: np.ndarray,
        mask: np.ndarray,
        metric: Union[str, Callable],
        n_jobs : Union[int, None] = -1):
    """
    Core iteration strategy for selecting another record from our unlabeled records.

    Given a set of labeled records (X_training) and unlabeled records (X_pool) with uncertainty scores (X_uncertainty),
    we'd like to identify the best instance in X_pool that best balances uncertainty and dissimilarity.

    Refer to Cardoso et al.'s "Ranked batch-mode active learning":
        https://www.sciencedirect.com/science/article/pii/S0020025516313949

    Args:
        X_training: Mix of both labeled and unlabeled records.
        X_pool: Unlabeled records to be selected for labeling.
        X_training_feat: feature vectors for the training data
        X_pool_feat: feature vectors for the unlabeld data
        X_uncertainty: Uncertainty scores for unlabeled records to be selected for labeling.
        mask: Mask to exclude previously selected instances from the pool.
        metric: This parameter is passed to :func:`~sklearn.metrics.pairwise.pairwise_distances`.
        n_jobs: This parameter is passed to :func:`~sklearn.metrics.pairwise.pairwise_distances`.

    Returns:
        Index of the best index from X chosen to be labelled; a single record from our unlabeled set that is considered
        the most optimal incremental record for including in our query set.
    """
    # Extract the number of labeled and unlabeled records.
    n_labeled_records = X_training.shape[0]
    n_unlabeled = X_pool[mask].shape[0]

    # Determine our alpha parameter as |U| / (|U| + |D|). Note that because we
    # append to X_training and remove from X_pool within `ranked_batch`,
    # :alpha: is not fixed throughout our model's lifetime.
    alpha = n_unlabeled / (n_unlabeled + n_labeled_records)

    # Compute pairwise distance (and then similarity) scores from every unlabeled record
    # to every record in X_training. The result is an array of shape (n_samples, ).
    
    ################## TODO: replace this part with a better similarity computation ############################
    '''
    Args:
        X_u: unlabeled data
        X_l: labeled data
    Returns:
        pairwise distance between the two points
    '''
    if X_pool_feat is None or X_training_feat is None:
        X_pool_features = X_pool[mask].reshape((len(X_pool[mask]), -1))
        X_training_features = X_training.reshape((len(X_training), -1))
    else:
        X_pool_features = X_pool_feat[mask]
        X_training_features = X_training_feat
        
    if n_jobs == 1 or n_jobs is None:
        _, distance_scores = pairwise_distances_argmin_min(X_pool_features, X_training_features, metric=metric)
    else:
        distance_scores = pairwise_distances(X_pool_features, X_training_features, metric=metric, n_jobs=n_jobs).min(axis=1)
    ############################################################################################################
    
    similarity_scores = 1 / (1 + distance_scores)

    # Compute our final scores, which are a balance between how dissimilar a given record
    # is with the records in X_uncertainty and how uncertain we are about its class.
    scores = alpha * (1 - similarity_scores) + (1 - alpha) * X_uncertainty[mask]

    # Isolate and return our best instance for labeling as the one with the largest score.
    best_instance_index_in_unlabeled = np.argmax(scores)
    n_pool, *rest = X_pool.shape
    unlabeled_indices = [i for i in range(n_pool) if mask[i]]
    best_instance_index = unlabeled_indices[best_instance_index_in_unlabeled]
    mask[best_instance_index] = 0
    return best_instance_index, np.expand_dims(X_pool[best_instance_index], axis=0), mask
    
#    best_instance_index = np.argmax(scores)
#    mask[best_instance_index] = 0
#    return best_instance_index, X_pool[best_instance_index], mask


def ranked_batch(classifier,
                 labeled,
                 unlabeled,
                 X_training_feat,
                 X_pool_feat,
                 uncertainty_scores: np.ndarray,
                 n_instances: int,
                 metric: Union[str, Callable],
                 n_jobs: Union[int, None]) -> np.ndarray:
    """
    Query our top :n_instances: to request for labeling.

    Refer to Cardoso et al.'s "Ranked batch-mode active learning":
        https://www.sciencedirect.com/science/article/pii/S0020025516313949

    Args:
        classifier: active learner supported active learning models.
        labeled: the labeled dataset
        unlabeled: Set of records to be considered for our active learning model.
        X_training_feat: feature vectors of the labeled dataset
        X_pool_feat: feature vectors of the unlabeled dataset
        uncertainty_scores: Our classifier's predictions over the response variable.
        n_instances: Limit on the number of records to query from our unlabeled set.
        metric: This parameter is passed to :func:`~sklearn.metrics.pairwise.pairwise_distances`.
        n_jobs: This parameter is passed to :func:`~sklearn.metrics.pairwise.pairwise_distances`.

    Returns:
        The indices of the top n_instances ranked unlabelled samples.
    """
    # Define our record container and the maximum number of records to sample.
    instance_index_ranking = []
    ceiling = np.minimum(unlabeled.shape[0], n_instances)

    # mask for unlabeled initialized as transparent
    mask = np.ones(unlabeled.shape[0], np.bool)

    for _ in tqdm(range(ceiling)):

        # Receive the instance and corresponding index from our unlabeled copy that scores highest.
        instance_index, instance, mask = select_instance(X_training=labeled, 
                                                         X_pool=unlabeled,
                                                         X_training_feat=X_training_feat,
                                                         X_pool_feat=X_pool_feat,
                                                         X_uncertainty=uncertainty_scores, 
                                                         mask=mask,
                                                         metric=metric, 
                                                         n_jobs=n_jobs)

        # Add our instance we've considered for labeling to our labeled set. Although we don't
        # know it's label, we want further iterations to consider the newly-added instance so
        # that we don't query the same instance redundantly.
        labeled = np.concatenate((labeled, instance))
        X_training_feat = np.concatenate((X_training_feat, np.array([X_pool_feat[instance_index]])))

        # Finally, append our instance's index to the bottom of our ranking.
        instance_index_ranking.append(instance_index)

    # Return numpy array, not a list.
    return np.array(instance_index_ranking)


def uncertainty_batch_sampling(model: Model,
                               X_u: Union[np.ndarray, sp.csr_matrix],
                               features_labeled = None,
                               features_unlabeled = None,
                               n_instances: int = 20,
                               metric: Union[str, Callable] = 'euclidean',
                               n_jobs: Optional[int] = None,
                               **uncertainty_measure_kwargs
                               ) -> Tuple[np.ndarray, Union[np.ndarray, sp.csr_matrix]]:
    """
    Batch sampling query strategy. Selects the least sure instances for labelling.

    Refer to Cardoso et al.'s "Ranked batch-mode active learning":
        https://www.sciencedirect.com/science/article/pii/S0020025516313949

    Args:
        classifier: active learning model.
        X: Set of records to be considered for our active learning model.
        features_labeled: feature vectors of the labeled data to be used for similarity matrix computation.
        features_unlabeled: feature vectors of the unlabeled data to be used for similarity matrix computation.
        n_instances: Number of records to return for labeling from `X`.
        metric: This parameter is passed to :func:`~sklearn.metrics.pairwise.pairwise_distances`
        n_jobs: If not set, :func:`~sklearn.metrics.pairwise.pairwise_distances_argmin_min` is used for calculation of
            distances between samples. Otherwise it is passed to :func:`~sklearn.metrics.pairwise.pairwise_distances`.
        **uncertainty_measure_kwargs: Keyword arguments to be passed for the :meth:`predict_proba` of the classifier.

    Returns:
        Indices of the instances from `X` chosen to be labelled; records from `X` chosen to be labelled.
    """
    uncertainty = segmentation_uncertainty(model, X_u, **uncertainty_measure_kwargs)
    query_indices = ranked_batch(model, 
                                 labeled=model.X_training, 
                                 unlabeled=X_u,
                                 X_training_feat = features_labeled,
                                 X_pool_feat = features_unlabeled,
                                 uncertainty_scores=uncertainty,
                                 n_instances=n_instances, 
                                 metric=metric, 
                                 n_jobs=n_jobs)
    return query_indices, X_u[query_indices]

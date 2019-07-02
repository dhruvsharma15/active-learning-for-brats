#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:06:41 2019

@author: dhruv.sharma
"""

import keras.backend as K
from keras.callbacks import  ModelCheckpoint,Callback
import tensorflow as tf 

import numpy as np
from strategies.uncertainty import uncertainty_sampling

class SGDLearningRateTracker(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.get_value(optimizer.lr)
        decay = K.get_value(optimizer.decay)
        lr=lr/2
        decay=decay*2
        K.set_value(optimizer.lr, lr)
        K.set_value(optimizer.decay, decay)
        print('LR changed to:',lr)
        print('Decay changed to:',decay)


class ActiveLearner():
    '''
    This class defines our Active Learning module. It combines with the model to provide the 
    AL framework for image segmentation task.
    Args:
        model: the segmentation model being used. Should be an instance of Keras.Model
        query_strategy: a callable function for the query strategy to be used
        X_training: the pool of the training data
        y_training: labels for the training data
        **fit_kwargs: keyword arguments for the git function
    '''
    def __init__(self,
                model,
                X_training,
                y_training,
                X_val = None,
                y_val = None,
                query_strategy=uncertainty_sampling,
                **fit_kwargs
                ):
        assert callable(query_strategy), 'query_strategy must be callable'
        
        self.model = model
        self.query_strategy = query_strategy
        
        self.X_training = X_training
        self.y_training = y_training
        
        self.X_val = X_val
        self.y_val = y_val
        
        if self.X_training is not None:
            self._fit_to_known(**fit_kwargs)
    
    def _add_training_data(self, X, y):
        """
        Adds the new data and label to the known data, but does not retrain the model.

        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.

        Note:
            If the classifier has been fitted, the features in X have to agree with the training samples which the
            classifier has seen.
        """
        
        if self.X_training is None:
            self.X_training = X
            self.y_training = y
        else:
            try:
                self.X_training = np.concatenate((self.X_training, X))
                self.y_training = np.concatenate((self.y_training, y))
            except ValueError:
                raise ValueError('the dimensions of the new training data and label must'
                                 'agree with the training data and labels provided so far')

    def _fit_to_known(self, **fit_kwargs):
        """
        Fits self.model to the training data and labels provided to it so far.

        Args:
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.

        Returns:
            self
        """
        

        K.get_session().close()
        K.set_session(tf.Session())
        K.get_session().run(tf.global_variables_initializer())

        checkpointer = ModelCheckpoint(filepath='trained_weights_AL3/ResUnet.{epoch:02d}.hdf5', verbose=1)
        validation_data = None
        if(self.X_val is not None and self.y_val is not None):
            validation_data = (self.X_val, self.y_val)
        self.model.fit(self.X_training, self.y_training, validation_data = validation_data, callbacks = [checkpointer,SGDLearningRateTracker()], **fit_kwargs)

        return self

    def _fit_on_new(self, X, y, **fit_kwargs):
        """
        Fits self.model to the given data and labels.

        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.

        Returns:
            self
        """
        checkpointer = ModelCheckpoint(filepath='trained_weights/ResUnet.{epoch:02d}.hdf5', verbose=1)
        validation_data = None
        if(self.X_val is not None and self.y_val is not None):
            validation_data = (self.X_val, self.y_val)
        self.model.fit(X, y, validation_data = validation_data, callbacks = [checkpointer,SGDLearningRateTracker()], **fit_kwargs)
      
        return self

    def fit(self, X, y, **fit_kwargs):
        """
        Interface for the fit method of the predictor. Fits the predictor to the supplied data, then stores it
        internally for the active learning loop.

        Args:
            X: The samples to be fitted.
            y: The corresponding labels.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.

        Returns:
            self
        """
#        self.X_training, self.y_training = X, y
        return self._fit_to_known(**fit_kwargs)

    def predict(self, X, **predict_kwargs):
        """
        model predictions for X. Interface with the predict method of the estimator.

        Args:
            X: The samples to be predicted.
            **predict_kwargs: Keyword arguments to be passed to the predict method of the estimator.

        Returns:
            model predictions for X.
        """
        return self.model.predict(X, **predict_kwargs)
    
    def evaluate(self, X, y, model_path=None, **evalute_kwargs):
        """
        model evaluation for X. Interface with the evaluate method of the estimator.

        Args:
            X: The samples to be evaluated.
            y: ground truth for the samples.
            model_path: path for weights to be restored
            **evaluate_kwargs: Keyword arguments to be passed to the evaluate method of the estimator.

        Returns:
            model metrics after evaluation of X.
        """
        if model_path is not None:
            self.model.load_weights(model_path)
        return self.model.evaluate(X, y, **evalute_kwargs)
        
    
    def query(self, *query_args, **query_kwargs):
        """
        Finds the n_instances most informative point in the data provided by calling the query_strategy function.

        Args:
            *query_args: The arguments for the query strategy. For instance, in the case of
                :func:`~uncertainty.uncertainty_sampling`, it is the pool of samples from which the query strategy
                should choose instances to request labels.
            **query_kwargs: Keyword arguments for the query strategy function.

        Returns:
            Value of the query_strategy function. Should be the indices of the instances from the pool chosen to be
            labelled and the instances themselves. Can be different in other cases, for instance only the instance to be
            labelled upon query synthesis.
        """
        query_result = self.query_strategy(self, *query_args, **query_kwargs)
        return query_result
    
    def teach(self, X, y, only_new = False, **fit_kwargs):
        """
        Adds X and y to the known training data and retrains the predictor with the augmented dataset.

        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.
            only_new: If True, the model is retrained using only X and y, ignoring the previously provided examples.
                Useful when working with models where the .fit() method doesn't retrain the model from scratch (e. g. in
                tensorflow or keras).
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        self._add_training_data(X, y)
        if not only_new:
            self._fit_to_known(**fit_kwargs)
        else:
            self._fit_on_new(X, y, **fit_kwargs)
  
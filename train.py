from __future__ import print_function
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# 
## The GPU id to use, usually either "0" or "1";
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

from datetime import datetime
import numpy as np
from functools import partial
import configparser

import keras.backend as K
import tensorflow as tf
from keras.models import Model

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from active_learner import ActiveLearner
from model import Unet_model
from strategies.uncertainty import *
from strategies.coreset_ranked_sampling import informative_batch_sampling
from strategies.batch_sampling import uncertainty_batch_sampling

config = tf.ConfigProto(intra_op_parallelism_threads=8,
                        inter_op_parallelism_threads=8,
                        allow_soft_placement=True,
                        device_count = {'CPU': 8})
session = tf.Session(config=config)
K.set_session(session)

def main():
    start=datetime.now()
    
    ######################3 data paths ######################
    config = configparser.ConfigParser()
    config.read("config.ini")
    x_train_patches_path = config.get('Paths', 'X_train_patches_path')
    y_train_patches_path = config.get('Paths', 'y_train_patches_path')
    x_val_patches_path = config.get('Paths', 'X_val_patches_path')
    y_val_patches_path = config.get('Paths', 'y_val_patches_path')
    x_test_patches_path = config.get('Paths', 'X_test_patches_path')
    y_test_patches_path = config.get('Paths', 'y_test_patches_path')
    weights_path = config.get('Paths', 'weights_path')
    
    ###################### model hyperparameters ###########
    try:
        nb_initial_epochs = int(config.get('model_params', 'nb_initial_epochs'))
    except:
        nb_initial_epochs = 10
        
    try:
        nb_active_epochs = int(config.get('model_params', 'nb_active_epochs'))
    except:
        nb_active_epochs = 10
        
    try:
        batch_size = int(config.get('model_params', 'batch_size'))
    except:
        batch_size = 4
    
    ###################### active learning hyperparameters #####
    try:
        nb_labeled = int(config.get('AL_params', 'nb_labeled'))
    except:
        nb_labeled = 9000
        
    try:
        nb_iterations = int(config.get('AL_params', 'nb_iterations'))
    except:
        nb_iterations = 5
    
    try:
        nb_annotations = int(config.get('AL_params', 'nb_annotations'))
    except:
        nb_annotations = 600
        
    try:
        strategy = int(config.get('AL_params', 'query_strategy'))
        if(strategy == 'informative_batch_sampling'):
            query_strategy = partial(uncertainty_batch_sampling, n_instances=nb_annotations)
        if(strategy == 'uncertainty_batch_sampling'):
            query_strategy = partial(uncertainty_batch_sampling, n_instances=nb_annotations)
    except:
        query_strategy = partial(uncertainty_batch_sampling, n_instances=nb_annotations)
    
    # Data Loading
    Y_=np.load(x_train_patches_path, mmap_mode = 'r').astype(np.uint8)
    X_=np.load(y_train_patches_path, mmap_mode = 'r').astype(np.float32)
    
    Y_labels=Y_[:20000]
    X_patches=X_[:20000]
    del Y_, X_
    print("Data shape:",X_patches.shape)

    Y_valid = np.load(y_val_patches_path).astype(np.uint8)
    X_valid = np.load(x_val_patches_path).astype(np.float32)
    
    Y_test = np.load(y_test_patches_path).astype(np.uint8)
    X_test = np.load(x_test_patches_path).astype(np.float32)
    
    
    ##################################################
        
#    nb_unlabeled = X_patches.shape[0] - nb_labeled
    initial_idx = np.random.choice(range(len(X_patches)), size=nb_labeled, replace=False)

    ##################################################
    
    # DB definition
    

    X_labeled_train = X_patches[initial_idx]
    y_labeled_train = Y_labels[initial_idx]

    X_pool = np.delete(X_patches, initial_idx, axis=0)
    y_pool = np.delete(Y_labels, initial_idx, axis=0)
    # (1) Initialize model
    
    unet = Unet_model(img_shape=(128,128,4))
    model = unet.model
    
        
    # Active learner initilization
    learner = ActiveLearner(model = model,
                            query_strategy = query_strategy,
                            X_training = X_labeled_train,
                            y_training = y_labeled_train,
                            weights_path = weights_path,
                            X_val = X_valid,
                            y_val = Y_valid,
                            verbose = 1, epochs = nb_initial_epochs,
                            batch_size = batch_size
                            )
    ############ testing ############
    val_output = learner.evaluate(X_test, Y_test, verbose = 1)
        
    for i in range(len(learner.model.metrics_names)):
        print(learner.model.metrics_names[i], val_output[i])
    ################################
    
    layer_name = 'conv2d_11'
    intermediate_layer_model = Model(inputs=learner.model.input,
                                         outputs=learner.model.get_layer(layer_name).output)
    for idx in range(nb_iterations):
        nb_active_epochs = nb_active_epochs + 2
        ## features of the labeled and the unlabeled pool ############
        print('extracting features from the encoder of the UNet')
        n_dims = min(1024,min(len(learner.X_training), len(X_pool)))
        
        print('applying PCA for labeled pool')
        labeled_inter = intermediate_layer_model.predict(learner.X_training)
        labeled_inter = labeled_inter.reshape((len(labeled_inter), -1))
        labeled_inter = StandardScaler().fit_transform(labeled_inter)
        
        print('applying PCA for unlabeled pool')
        unlabeled_inter = intermediate_layer_model.predict(X_pool)
        unlabeled_inter = unlabeled_inter.reshape((len(unlabeled_inter), -1))
        unlabeled_inter = StandardScaler().fit_transform(unlabeled_inter)
        
        pca = PCA(n_components = min(n_dims, min(labeled_inter.shape)))
        features_labeled = pca.fit_transform(labeled_inter)
        
        pca = PCA(n_components = min(n_dims, min(unlabeled_inter.shape)))
        features_unlabeled = pca.fit_transform(unlabeled_inter)
        #################################################################
        print('Query no. %d' % (idx + 1))
        print('Training data shape', learner.X_training.shape)
        print('Unlabeled data shape', X_pool.shape)
        query_idx, query_instance = learner.query(X_u=X_pool, n_instances = nb_annotations, 
                                                  features_labeled = features_labeled, 
                                                  features_unlabeled = features_unlabeled)
        
        learner.teach(
            X=X_pool[query_idx], y=y_pool[query_idx], only_new=False,
            verbose=1, epochs = nb_active_epochs, batch_size = batch_size
        )
        # remove queried instance from pool
        print("patches annotated: ", X_pool[query_idx].shape)
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)
        
        print('Training data shape after this query', learner.X_training.shape)
        print('Unlabeled data shape after this query', X_pool.shape)
        
        ####### testing ##############
        val_output = learner.evaluate(X_test, Y_test, verbose=1)
        
        for i in range(len(learner.model.metrics_names)):
            print(learner.model.metrics_names[i], val_output[i])
        
        del labeled_inter, unlabeled_inter, features_labeled, features_unlabeled
        
    ############### testing #######################
    model_path = './trained_weights/'
    model_weight_paths = sorted(os.listdir(model_path))[-1]
    
    print('testing the model')
    
    weights_path = model_path + model_weight_paths

    val_output = learner.evaluate(X = X_test, y = Y_test, model_path=weights_path, batch_size=4, verbose=1)
    
    for i in range(len(learner.model.metrics_names)):
        print(learner.model.metrics_names[i], val_output[i])
    
    print('time taken to run the code ', datetime.now()-start)
    
if __name__ == '__main__':
    main()
from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from datetime import datetime
import numpy as np
from functools import partial

from keras.utils import np_utils
import keras.backend as K
import tensorflow as tf
from keras.models import Model

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from active_learner import ActiveLearner
from model import Unet_model
from strategies.uncertainty import *
from strategies.batch_sampling import uncertainty_batch_sampling

config = tf.ConfigProto(intra_op_parallelism_threads=8,
                        inter_op_parallelism_threads=8,
                        allow_soft_placement=True,
                        device_count = {'CPU': 8})
session = tf.Session(config=config)
K.set_session(session)

def main():
    start=datetime.now()
    # Data Loading and Preprocessing
    
    Y_labels=np.load("../Brats_patches_data/y_train_small.npy").astype(np.uint8)
    X_patches=np.load("../Brats_patches_data/x_train_small.npy").astype(np.float32)
    print("Data shape:",X_patches.shape)
#    X_train = X_patches[:100]
#    X_test = X_patches[100:]
    
    Y_valid = np.load("../Brats_patches_data/y_val.npy").astype(np.uint8)
    X_valid = np.load("../Brats_patches_data/x_val.npy").astype(np.float32)
    
    Y_test = np.load("../Brats_patches_data/y_test.npy").astype(np.uint8)
    X_test = np.load("../Brats_patches_data/x_test.npy").astype(np.float32)
    
    
    ##################################################
        
    nb_labeled = 2000
#    nb_unlabeled = X_patches.shape[0] - nb_labeled
    initial_idx = np.random.choice(range(len(X_patches)), size=nb_labeled, replace=False)

    nb_iterations = 10
    nb_annotations = 500

    
    nb_initial_epochs = 10
    nb_active_epochs = 10
    batch_size = 4

    ##################################################
    
    # DB definition
    
#    y_train = Y_labels[:100]
#    y_test = Y_labels[100:]
    
    X_labeled_train = X_patches[initial_idx]
    y_labeled_train = Y_labels[initial_idx]

    X_pool = np.delete(X_patches, initial_idx, axis=0)
    y_pool = np.delete(Y_labels, initial_idx, axis=0)
    # (1) Initialize model
    
    unet = Unet_model(img_shape=(128,128,4))
    model = unet.model
    
        
    # Active loop
    preset_batch = partial(uncertainty_batch_sampling, n_instances=nb_annotations, n_jobs = 8)

    learner = ActiveLearner(model = model,
                            query_strategy = preset_batch,
                            X_training = X_labeled_train,
                            y_training = y_labeled_train,
                            X_val = X_valid,
                            y_val = Y_valid,
                            verbose = 1, epochs = nb_initial_epochs,
                            batch_size = batch_size
                            )
    ############ testing ############
    val_output = learner.evaluate(X_test, Y_test)
        
    for i in range(len(learner.model.metrics_names)):
        print(learner.model.metrics_names[i], val_output[i])
    ################################
    
    for idx in range(nb_iterations):
        nb_active_epochs = nb_active_epochs + 2
        ## features of the labeled and the unlabeled pool ############
        print('extracting features from the encoder of the UNet')
        
        layer_name = 'conv2d_11'
        n_dims = min(1024,min(len(learner.X_training), len(X_pool)))
        intermediate_layer_model = Model(inputs=learner.model.input,
                                         outputs=learner.model.get_layer(layer_name).output)
        print('applying PCA for labeled pool')
        labeled_inter = intermediate_layer_model.predict(learner.X_training)
        labeled_inter = labeled_inter.reshape((len(labeled_inter), -1))
        labeled_inter = StandardScaler().fit_transform(labeled_inter)
        
        pca = PCA(n_components = min(n_dims, min(labeled_inter.shape)))
        features_labeled = pca.fit_transform(labeled_inter)
        
        print('applying PCA for unlabeled pool')
        unlabeled_inter = intermediate_layer_model.predict(X_pool)
        unlabeled_inter = unlabeled_inter.reshape((len(unlabeled_inter), -1))
        unlabeled_inter = StandardScaler().fit_transform(unlabeled_inter)
        
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
        
        ####### testing ##############
        val_output = learner.evaluate(X_test, Y_test)
        
        for i in range(len(learner.model.metrics_names)):
            print(learner.model.metrics_names[i], val_output[i])
        
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
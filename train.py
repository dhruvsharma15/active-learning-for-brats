from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
from functools import partial

from keras.utils import np_utils

from active_learner import ActiveLearner
from unet import Unet_model
from strategies.uncertainty import *
from strategies.batch_sampling import uncertainty_batch_sampling

def main():
    
    # Data Loading and Preprocessing
    
    Y_labels=np.load("../Brats_patches_data/y_train.npy").astype(np.uint8)
    X_patches=np.load("../Brats_patches_data/x_train.npy").astype(np.float32)
    model_to_load=None
    print("Data shape:",X_patches.shape)
#    X_train = X_patches[:100]
#    X_test = X_patches[100:]
    
    Y_valid = np.load("../Brats_patches_data/y_val.npy").astype(np.uint8)
    X_valid = np.load("../Brats_patches_data/x_val.npy").astype(np.float32)
    
    
    ##################################################
    
    # CEAL params
    
    nb_labeled = 200
#    nb_unlabeled = X_patches.shape[0] - nb_labeled
    initial_idx = np.random.choice(range(len(X_patches)), size=nb_labeled, replace=False)

    nb_iterations = 8
    nb_annotations = 100

    
    nb_initial_epochs = 5
    nb_active_epochs = 5
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
    model = unet.compile_unet()
    model.load_weights(model_to_load)
        
    # Active loop
    preset_batch = partial(uncertainty_batch_sampling, n_instances=nb_annotations)

    learner = ActiveLearner(model = model,
                            query_strategy = preset_batch,
                            X_training = X_labeled_train,
                            y_training = y_labeled_train,
                            X_val = X_valid,
                            y_val = Y_valid,
                            verbose = 1, epochs = nb_initial_epochs,
                            batch_size = batch_size
                            )
    
    for idx in range(nb_iterations):
        print('Query no. %d' % (idx + 1))
        query_idx, query_instance = learner.query(X_u=X_pool)
        
        learner.teach(
            X=X_pool[query_idx], y=y_pool[query_idx], only_new=True,
            verbose=1, epochs = nb_active_epochs, batch_size = batch_size
        )
        # remove queried instance from pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)
    
if __name__ == '__main__':
    main()
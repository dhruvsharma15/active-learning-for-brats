from __future__ import print_function

import numpy as np

from keras.utils import np_utils

from active_learner import ActiveLearner
from unet import Unet_model
from strategies.uncertainty import *
from strategies.batch_sampling import *

def main():
    
    # Data Loading and Preprocessing
    
    """ BraTS data:
        Train Set = 60,000 samples
        Validation Set = 10,000 samples """
    
    Y_labels=np.load("../Brats_patches_data/y_dataset_first_part.npy").astype(np.uint8)[:150]
    X_patches=np.load("../Brats_patches_data/x_dataset_first_part.npy").astype(np.float32)[:150]
    
    X_train = X_patches[:100]
    X_test = X_patches[100:]
    
    
    ##################################################
    
    # CEAL params
    
    nb_labeled = 20
    nb_unlabeled = X_train.shape[0] - nb_labeled
    
    nb_iterations = 8
    nb_annotations = 10

    
    nb_initial_epochs = 2
    nb_active_epochs = 2
    batch_size = 4

    ##################################################
    
    # DB definition
    
    y_train = Y_labels[:100]
    y_test = Y_labels[100:]
    
    X_labeled_train = X_train[0:nb_labeled]
    y_labeled_train = y_train[0:nb_labeled]
    X_pool = X_train[nb_labeled:]
    y_pool = y_train[nb_labeled:]
    
    # (1) Initialize model
    
    unet = Unet_model(img_shape=(128,128,4))
    model = unet.compile_unet()
        
    # Active loop
    learner = ActiveLearner(model = model,
                            query_strategy = uncertainty_sampling,
                            X_training = X_labeled_train,
                            y_training = y_labeled_train,
                            verbose = 1, epochs = nb_initial_epochs,
                            batch_size = batch_size
                            )
    
    for idx in range(nb_iterations):
        print('Query no. %d' % (idx + 1))
        query_idx, query_instance = learner.query(X_pool, n_instances = nb_annotations)
        
        learner.teach(
            X=X_pool[query_idx], y=y_pool[query_idx], only_new=True,
            verbose=1, epochs = nb_active_epochs, batch_size = batch_size
        )
        # remove queried instance from pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)
    
if __name__ == '__main__':
    main()
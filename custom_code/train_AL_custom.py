from __future__ import print_function

import os
from datetime import datetime
from time import time
import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils

from unet import Unet_model
from utils import *
from uncertainty import *
from batch_sampling import *

def train_loop(X_train, y_train, X_test, y_test, nb_epochs, batch_size, iteration, log_file):
    for current_epoch in range(0, nb_epochs):
        print("Number of epoch: " + str(current_epoch + 1) + "/" + str(nb_epochs))

        model.fit(X_train, y_train, batch_size=batch_size,
                  epochs=1, validation_data=(X_test, y_test),
                  verbose=1)  # <-- pensar en sets de validacio augmentables

        score_train = model.evaluate(X_labeled_train, y_labeled_train, verbose=0)
        score_test = model.evaluate(X_test, y_test, verbose=0)

        log_file.write('{0} {1} {2} {3} {4} {5} {6} \n'.format(str(iteration), str(current_epoch + 1),
                                                               str(len(X_train)), str(score_train[0]),
                                                               str(score_train[1]), str(score_test[0]),
                                                               str(score_test[1])))


initial_time = time()
# Paths

model_name = "ceal_unet"
model_path = "models/" + model_name + "/"
logs_path = model_path + "/logs/"
weights_path = "models/" + model_name + "/weights/"

if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    print('Path created: ', logs_path)

if not os.path.exists(weights_path):
    os.makedirs(weights_path)
    print('Path created: ', weights_path)

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
initial_decay_rate = 0.6
decay_rate = 0.5
thresh = None

nb_initial_epochs = 2
nb_active_epochs = 2
batch_size = 4
nb_classes = 10

##################################################

# DB definition

y_train = Y_labels[:100]
y_test = Y_labels[100:]

X_labeled_train = X_train[0:nb_labeled]
y_labeled_train = y_train[0:nb_labeled]
X_unlabeled_train = X_train[nb_labeled:]

# (1) Initialize model
iteration = 0

unet = Unet_model(img_shape=(128,128,4))
model = unet.compile_unet()

data = datetime.now()
f = open(logs_path + model_name + "_log" + str(data.month) + str(data.day) + str(data.hour) + str(data.minute),
         'a')
f.write("Log format: iteration / epoch / nb_train / loss_train / acc_train / loss_test / acc_test\n"
        "iteration: 0 - Initialization / ~ 0 - Active training\n\n")

train_loop(X_labeled_train, y_labeled_train, X_test, y_test, nb_initial_epochs, batch_size, iteration, f)


# Active loop

for iteration in range(1, nb_iterations + 1):
    # (2) Labeling

    print("Getting predictions...")
    t = time()
    predictions = model.predict(X_unlabeled_train, verbose=0)
    print("Time elapsed: " + str(time() - t) + " s")
    
#    uncertain_idx, uncertain_samples  = uncertainty_sampling(model, X_unlabeled_train, n_instances=nb_annotations)
    uncertain_idx, uncertain_samples = uncertainty_batch_sampling(model, X_unlabeled_train, 
                                                                  X_labeled_train, n_instances = nb_annotations)
        
    y_oracle_train = y_train[uncertain_idx]
    X_oracle_train = X_unlabeled_train[uncertain_idx]
    np.delete(X_unlabeled_train, uncertain_idx)

    # (3) Training
    train_loop(X_oracle_train, y_oracle_train, X_test, y_test, nb_active_epochs, batch_size, iteration, f)
    X_labeled_train = np.concatenate((X_labeled_train, X_oracle_train))
    y_labeled_train = np.concatenate((y_labeled_train, y_oracle_train))

print("Time elapsed: " + str(time() - initial_time) + " s")
f.close()

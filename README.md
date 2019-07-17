# Applying active learning for multimodal brain tumor segmentation

### Brief overview

This repository provides source code for a deep convolutional neural network architecture designed for brain tumor segmentation with BraTS2018 dataset by involving Active Learning in it. 
The architecture is fully convolutional network (FCN) built upon the well-known U-net model and it makes use of residual units instead of plain units to speedup training and convergence.
The Brain tumor segmentation problem exhibits severe class imbalance where the healthy voxels comprise 98% of total voxels,0.18% belongs to necrosis ,1.1% to edema and non-enhanced and 0.38% to enhanced tumor. 
The issue is addressed by: 1) adopting a patch-based training approach; 2) using a custom loss function that accounts for the imbalance. 
During training, 2D patches of size 128x128 from the axial plane are randomly sampled. And by doing so it allows to dismiss patches from pixels with zero intensity and therefore it helps a bit to alleviate the problem.

### Active Learning
Active Learning is the algorithm to intelligently select data points for training the model. This ensures that we are using limited resources for getting the data annotated. To select the points intelligently, we need to find the informativeness of the data point - uncertainty and representativeness. These are captures using the query strategies. This module implements 2 query strategies:
* Uncertainty sampling
* ranked batch-mode sampling 

The implementation is based on keras and tested on Tensorflow backends.

### Requirements

To run the code, you first need to install the following prerequisites: 

* Python 3.5 or above
* numpy
* keras
* tensorflow-gpu==1.13.1
* scipy
* SimpleITK
* tqdm

### How to run

Git clone this repository in the same directory as your BraTS data - `Brats2018_training`.

1. Execute first `extract_patches.py` to prepare the training, validation, and testing datasets which will be saved in the `Brats_patches_data` directory. This also segregates the images into three different folders, one for each split, which can later be utilized for generating the predictions stored under the folder of `data_split`.
2. then `train.py` to train the model. The trained weights will be saved in the directory `brain_segmentation`.
3. Use `predict.py` to run prediction on the dataset. Define the path to folder where the dataset to be predicted is stored. 

```
python extract_patches.py
python train.py
python predict.py
```
Running `predict.py` generates the segmentation mask and also the metrics for each image. For example:
```
Volume ID:  HGG/Brats18_TCIA04_149_1
155/155 [==============================] - 3s 22ms/step
************************************************************
Dice complete tumor score : 0.7530
Dice core tumor score (tt sauf vert): 0.9235
Dice enhancing tumor score (jaune):0.8708 
**********************************************
Sensitivity complete tumor score : 0.9778
Sensitivity core tumor score (tt sauf vert): 0.9683
Sensitivity enhancing tumor score (jaune):0.9550 
***********************************************
Specificity complete tumor score : 0.9951
Specificity core tumor score (tt sauf vert): 0.9993
Specificity enhancing tumor score (jaune):0.9989 
***********************************************
Hausdorff complete tumor score : 8.0623
Hausdorff core tumor score (tt sauf vert): 4.5826
Hausdorff enhancing tumor score (jaune):9.2195 
***************************************************************
```

To use one of the query strategies:
1. Import the strategy module in your code and understand its documentation to know what it takes as input and what it returns.
2. Pass the parameters appropriately.

Sample of how to use the Active Learning framework with your model:

Make the neccesary imports and define the dataset. Uncertainty sampling query strategy is used here.
```
from active_learner import ActiveLearner
from strategies.uncertainty import *
import numpy as np

Y_labels = #... labels ...
X_patches = #... data points ...
```

Define the hyperparameters and define the labeled and the unlabeled pool of data.
```
nb_labeled = 2000

initial_idx = np.random.choice(range(len(X_patches)), size=nb_labeled, replace=False)

nb_iterations = 10
nb_annotations = 500


nb_initial_epochs = 10
nb_active_epochs = 10
batch_size = 4

X_labeled_train = X_patches[initial_idx]
y_labeled_train = Y_labels[initial_idx]

X_pool = np.delete(X_patches, initial_idx, axis=0)
y_pool = np.delete(Y_labels, initial_idx, axis=0)

model = #your keras model
```

Active Learning loop
```
learner = ActiveLearner(model = model,
		        query_strategy = uncertainty_sampling,
		        X_training = X_labeled_train,
		        y_training = y_labeled_train,
		        verbose = 1, epochs = nb_initial_epochs,
		        batch_size = batch_size
		        )


for idx in range(nb_iterations):
	print('Query no. %d' % (idx + 1))
	print('Training data shape', learner.X_training.shape)
	print('Unlabeled data shape', X_pool.shape)
	query_idx, query_instance = learner.query(X_u=X_pool, n_instances = nb_annotations)

	learner.teach(
	    X=X_pool[query_idx], y=y_pool[query_idx], only_new=False,
	    verbose=1, epochs = nb_active_epochs, batch_size = batch_size
	)
	# remove queried instance from pool
	X_pool = np.delete(X_pool, query_idx, axis=0)
	y_pool = np.delete(y_pool, query_idx, axis=0)
        
```



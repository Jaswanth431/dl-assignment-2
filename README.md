# DL-Assignment-2
# CNN Training with PyTorch(PART A)
## Overview
This repository contains code for training a Convolutional Neural Network (CNN) using PyTorch. The CNN model is trained on a custom dataset, and various hyperparameters can be configured through command-line arguments. The training progress and results are logged using Weights & Biases (wandb).

## How to run train_a.py
### Requirements
- Python 3.x
- PyTorch
- wandb
- torchvision
- matplotlib
- numpy
- argparse

### Prepare the Dataset:
- Organize your dataset into two folders: one for training images and one for validation images.
- Specify the paths to the training and testig image folders using the -td and -vd command-line arguments, respectively.

### Train the Model
Run the ```train_a.py``` script with the desired configuration options. Here's an example command:
```
python train_a.py [-h] [-wp WAND_PROJECT] [-e EPOCHS] [-lr LEARNING_RATE] [-b BATCH_SIZE] [-nf NUM_OF_FILTER] [-fs FILTER_SIZE [FILTER_SIZE ...]] [-af ACTV_FUNC] [-fm FILTER_MULTIPLIER] [-da] [-bn] [-do DROPOUT] [-ds DENSE_LAYER_SIZE] -td TRAIN_DIR -vd VAL_DIR
```

```
-e or --epochs: Default value is 10. Sets the number of epochs to train the neural network.
-lr or --learning_rate: Default value is 0.0001. Sets the learning rate used to optimize model parameters.
-b or --batch_size: Default value is 32. Specifies the batch size used for training.
-nf or --num_of_filter: Default value is 32. Sets the number of filters in the convolutional layers
-fs or --filter_size: Default value is [3, 3, 3, 3, 3]. Specifies the sizes of filters in the convolutional layers. The sizes are provided as a list separated by spaces.
-af or --actv_func: Default value is gelu. Chooses the activation function for the convolutional layers. Available options are gelu, silu, elu, and leaky_relu.
-fm or --filter_multiplier: Default value is 2. Specifies the filter multiplier for the convolutional layers. Available options are 0.5, 1, and 2.
-da or --data_augmentation: By default, data augmentation is turned off. To enable data augmentation, include -da in the command line.
-bn or --batch_normalization: By default, batch normalization is turned on. To disable batch normalization, include -bn in the command line.
-do or --dropout: Default value is 0.2. Sets the dropout rate for the fully connected layers.
-ds or --dense_layer_size: Default value is 256. Specifies the size of the dense layer.
-td or --train_dir: Specifies the folder containing training images. (Required)
-vd or --val_dir: Specifies the folder containing validation images. (Required)
```
### Example
```python train_a.py -wp myproject -e 10 -lr 0.0001 -b 32 -nf 32 -fs 3 3 3 3 3 -af gelu -fm 2 -da -bn -do 0.2 -ds 256 -td /path/to/train_data -vd /path/to/val_data```

# Fine-tuning ResNet50 Model with Hyperparameters(Part B)
## Overview
This Python script fine-tunes the ResNet50 model using specified hyperparameters. It allows for training the model with custom configurations and logs the training and validation accuracies to the Weights & Biases dashboard.

## How to run train_b.py
### Requirements
- Python 3.x
- PyTorch
- wandb
- torchvision
- matplotlib
- numpy
- argparse

### Prepare the Dataset:
- Organize your dataset into two folders: one for training images and one for validation images.
- Specify the paths to the training and testig image folders using the -td and -vd command-line arguments, respectively.

## Fine tune the model
```python train_b.py [-h] [-wp WAND_PROJECT] [-e EPOCHS] [-lr LEARNING_RATE] [-b BATCH_SIZE] [-lf LAST_UNFREEZE_LAYERS] -td TRAIN_DIR -vd VAL_DIR ```

```
-wp or --wandb_project: Specifies the project name used to track experiments in the Weights & Biases dashboard. Default is "DL proj".
-e or --epochs: Sets the number of epochs to train the neural network. Default is 10.
-lr or --learning_rate: Sets the learning rate used to optimize model parameters. Default is 0.0001.
-b or --batch_size: Specifies the batch size used for training. Default is 32.
-lf or --last_unfreeze_layers: Sets the number of layers from the last to unfreeze. Default is 1.
-td or --train_dir: Specifies the folder containing training images. (Required)
-vd or --val_dir: Specifies the folder containing validation images. (Required)
```

### Example
```
python train_b.py -e 15 -lr 0.001 -b 64 -lf 2 --train_dir /path/to/train_dir --val_dir /path/to/val_dir
```

  
## Running python notebooks
- They have to run cell by cell
- You can configure the hyperparameters in the ```h_params``` dictionary and then run all the cells until the cell with train function
- In part A, to run the sweeps, just run the last cell in dl-assignment-2a.ipynb notebook



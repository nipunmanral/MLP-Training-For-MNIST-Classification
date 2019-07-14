# Multilayer Perceptron Training for MNIST Classification

## Objective
This project aims to train a multilayer perceptron (MLP) deep neural network on MNIST dataset using numpy. The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) of handwritten digits has 784 input features (pixel values in each image) and 10 output classes representing numbers 0–9.

## Environment Setup
Download the codebase and open up a terminal in the root directory. Make sure python 3.6 is installed in the current environment. Then execute

    pip install -r requirements.txt

This should install all the necessary packages for the code to run.

## Dataset
The file mnist_testdata.hdf5 (not uploaded here due to size constraints) contains 10,000 images in the key ‘xdata’ and their corresponding labels in the key ‘ydata’. Each image is 784-dimensional and each label is one-hot 10-dimensional. So if the label for an image is [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], it means the image depicts a 3

## Code Information
Refer to the file `Different Configurations Report.pdf` for details and graphs on the various configurations of the MLP.

The best accuracy was obtained using the following configuration:
Input Layer - 784 neurons
1 Hidden layer of 512 neurons
Output Layer - 10 neurons
Batch Size – 50
Epoch – 50
Learning Rate - 0.09
Activation in Intermediate Layers – ReLu Activation
Activation in Output Layer – Softmax
Parameter Initialization – He Normalization
Final Training Accuracy : 99.99%
Final Test Accuracy : 98.26%

## Evaluation
Models are scored on classification accuracy which is simply the number of correct classifications divided by the total number of elements in the test set.

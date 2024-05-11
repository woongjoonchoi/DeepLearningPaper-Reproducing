# VGG Reproducing


## Introduction
VGG is a classical convolutional neural network architecture. It was based on an analysis of how to increase the depth of such networks. The network utilises small 3 x 3 filters. Otherwise the network is characterized by its simplicity: the only other components being pooling layers and a fully connected layer.

The code in this repo replicates the paper's configuration and trains the model from scratch with ImageNet data.  Techniques that were not available at the time (e.g. batchnormalization) were not used, configurations explicitly mentioned in the paper were not modified, and configurations not explicitly mentioned were set arbitrarily.

## 

# VGG Reproducing


## Introduction
VGG is a classical convolutional neural network architecture. It was based on an analysis of how to increase the depth of such networks. The network utilises small 3 x 3 filters. Otherwise the network is characterized by its simplicity: the only other components being pooling layers and a fully connected layer.

The code in this repo replicates the paper's configuration and trains the model from scratch with ImageNet data.  Techniques that were not available at the time (e.g. batchnormalization) were not used, configurations explicitly mentioned in the paper were not modified, and configurations not explicitly mentioned were set arbitrarily.

Due to limitations in GPU resources and time, training has only been possible up to about 15 epochs. If time or resources allow, training will be continued.

You can see many `ipynb` file in this repo. These files define the training configuration of vgg net and evaluate the evaluation methods step-by-step. Finally, the `.py` file contains all configuration and evaluation methods.

## Library Installation
```
torch==2.0.0+cu118
torchvision==0.15.1+cu118
albumentations==1.4.6
tqdm==4.66.2
wandb==0.16.6
```
Other libraries may be needed,  It's a bit of a hassle, but I hope you can install it on your own.If you post an issue regarding the insufficient library version, I will fix it. 
## Usage

How to train model 
```
python train.py --model_version [Spepcific version]
```
The model version includes [A,A_lrn,B,C,D,E].  
If you train your own model in gpu, you need least 12GB Gpu for batch size 64 .  

How to modify configuration
```
DatasetName = 'ImageNet' # Cifar  ,Cifar10, Mnist , ImageNet

## model configuration

```
In `config.py` , you can modify your configuration using python syntax.   
In particular, i support training on four datasets.`Cifar` , `Cifar10`, `Mnist`, `ImageNet` .  
If you want to train from ImageNet, you must download your dataset form [imagenet link](https://image-net.org/index.php).  ImageNet dataset require about 350GB disk storage for unzip . 
## todos

Define Your Own Dataset and add it to `dataset_class.py` .Then ,  train from scratch!.  

## Referenecs

[VGG paper](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjs_fuO5ISGAxUra_UHHc5GD6oQFnoECBUQAQ&url=https%3A%2F%2Farxiv.org%2Fabs%2F1409.1556&usg=AOvVaw17ak86ejVzNlyA2N-WpWmZ&opi=89978449)  
[karpathy/mingpt](https://github.com/karpathy/minGPT) The readme and project template were inspired by this repo.  
[grad accumulation](https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20?u=alband) i used grad accumulation code from this post.  
[torchvision vgg](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)  

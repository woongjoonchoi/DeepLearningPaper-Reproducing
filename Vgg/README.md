# VGG Reproducing


## Introduction
VGG is a classical convolutional neural network architecture. It was based on an analysis of how to increase the depth of such networks. The network utilises small 3 x 3 filters. Otherwise the network is characterized by its simplicity: the only other components being pooling layers and a fully connected layer.

The code in this repo replicates the paper's configuration and trains the model from scratch with ImageNet data.  Techniques that were not available at the time (e.g. batchnormalization) were not used, configurations explicitly mentioned in the paper were not modified, and configurations not explicitly mentioned were set arbitrarily.

Due to limitations in GPU resources and time, training has only been possible up to about 15 epochs. If time or resources allow, training will be continued.

You can see many `ipynb` file in this repo. These files define the training configuration of vgg net and evaluate the evaluation methods step-by-step. Finally, the `.py` file contains all configuration and evaluation methods.

## Several ineffective trials 
[Link](https://woongjoonchoi.github.io/Failure-with-vgg/) : Describes the process for resolving issues that occurred while training over 100 million models.  


When I first trained Vgg, I thought it would work well because it was a simple architecture. Because it was the first time training a model with more than 100 million parameters and a layer depth of more than 10 from scratch, i did not expect many issues in addition to convergence speed and accuracy issues. Therefore, i made several attempts to resolve these issues and found an optimized solution. I described the various attempts I made to find these solutions and how I came to this conclusion.
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

### How to train model 
```
python train.py --model_version [Spepcific version]
```
The model version includes [A,A_lrn,B,C,D,E].  
If you train your own model in gpu, you need least 12GB Gpu for batch size 64 .  

### How to modify configuration
```
DatasetName = 'ImageNet' # Cifar  ,Cifar10, Mnist , ImageNet

## model configuration

```
In `config.py` , you can modify your configuration using python syntax.   
In particular, i support training on four datasets.`Cifar` , `Cifar10`, `Mnist`, `ImageNet` .  
If you want to train from ImageNet, you must download your dataset form [imagenet link](https://image-net.org/index.php).  ImageNet dataset require about 350GB disk storage for unzip .  
There are many configurations other than those I mentioned in `config.py` . Because an explicit name is used in the configuration, and the python code is less than 1000 lines long, you can check it directly line by line.
## todos

Define Your Own Dataset and add it to `dataset_class.py` .Then ,  train from scratch!.  

## Noteworthy points

In the paper, it is said that after training model A, the weights of models B, C(vgg16), D(vgg19), and E(vgg19) were trained using the weights of model A.  
Then, after submitting the paper, they say they found a way to train B, C, D, and E without using transfer learning.  
It was said that xavier initialization was used as that method, but there is no description of the specific details.  
However, I succeeded in training models B, C, and D without transfer learning using xavier initialization.

## Expected Results
If i train for the time specified in the paper, it is expected that the results in the paper will be reproduced. 

## Single scale train and  evaluation
### Train dataset metric
|<img src="https://github.com/woongjoonchoi/DeepLearningPaper-Reproducing/assets/50165842/fbe18970-b815-48cd-bb0a-ebf2f03551ed"  width="300" height="300"> |<img src="https://github.com/woongjoonchoi/DeepLearningPaper-Reproducing/assets/50165842/0260d25a-f03a-4e72-999e-029ceed6a805"  width="300" height="300">| <img src="https://github.com/woongjoonchoi/DeepLearningPaper-Reproducing/assets/50165842/cdae299b-31a1-4338-a8a6-760fbe7a8658"  width="300" height="300">| 
|:--: |:--: |:--:  |
| *train/loss*  |*train/top-1-error* |*train/top-5-error*|

### Validation dataset metric

|<img src="https://github.com/woongjoonchoi/DeepLearningPaper-Reproducing/assets/50165842/bb21663b-3fb5-4272-9591-81122c12c6da"  width="300" height="300"> |<img src="https://github.com/woongjoonchoi/DeepLearningPaper-Reproducing/assets/50165842/f29d1202-425b-42c7-bfdc-85f0ea5fd2f4"  width="300" height="300">|<img src="https://github.com/woongjoonchoi/DeepLearningPaper-Reproducing/assets/50165842/c8d67e27-9559-4daf-aa21-f0149735e653"  width="300" height="300">| 
|:--: |:--: |:--:  |
| *val/loss*  |*val/top-1-error* |*val/top-5-error*|


### Vgg metric from ms research 
Plot from ResNet Paper. They trained 18 vgg on ImageNet Dataset from scratch .  
| <img src="https://github.com/woongjoonchoi/DeepLearningPaper-Reproducing/assets/50165842/a170125a-5ab7-4725-8e34-3f8853fa02d8"  width="300" height="300">| 
|:--: |
|*val/top-1-error* |

### Metric Table


|version |epoch|train S|test Q |val top-5 | val top-1|
|---|---|----|---|---|---|
|A |10 |256 |256| 30.85|57.49 |
|B | 10| 256|256 |32.01| 59.18|
|C |3 | 256| 256|68.87|87.11|
|D | |256 |256 || |
|Microsoft Research layer 18|10|-|-|-|**54.00**|


## Referenecs

[VGG paper](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjs_fuO5ISGAxUra_UHHc5GD6oQFnoECBUQAQ&url=https%3A%2F%2Farxiv.org%2Fabs%2F1409.1556&usg=AOvVaw17ak86ejVzNlyA2N-WpWmZ&opi=89978449)  
[karpathy/mingpt](https://github.com/karpathy/minGPT) The readme and project template were inspired by this repo.  
[grad accumulation](https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20?u=alband) i used grad accumulation code from this post.  
[torchvision vgg](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)  
[DeepLearning Book](https://www.deeplearningbook.org/)  I used Optimization solution from this book chp 8.  
[ResNet](https://arxiv.org/abs/1512.03385) I used Imagnet training plot  

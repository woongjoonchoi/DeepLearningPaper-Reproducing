import albumentations as A
import numpy as np


from torchvision.datasets import Caltech256 ,Caltech101 ,CIFAR100,CIFAR10,MNIST,ImageNet
import os
from PIL import Image

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


if __name__ =='__main__' :
    # print(os.getcwd())
    # imagenet_dataset = ImageNet(root='ImageNet',split='train')
    imagenet_dataset = ImageNet(root='ImageNet',split='train')
    # print('main')
    
    # label_dict = {}
    # for _,label in imagenet_dataset :
    #     if label not in label_dict:label_dict[label] =0
    #     label_dict[label]+=1
    # print(label_dict)
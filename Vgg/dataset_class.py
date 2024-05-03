import albumentations as A
import numpy as np


from torchvision.datasets import Caltech256 ,Caltech101 ,CIFAR100,CIFAR10,MNIST ,ImageNet
import os
from PIL import Image

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


class Custom_Caltech(Caltech256) :
    def __init__(self,root,transform=None,multi=False,s_max=None,s_min=256,train=True,val=False):
        self.S= None
        self.s_max = s_max
        self.s_min= s_min
        if multi :
            self.S = np.random.randint(low=self.s_min,high=self.s_max)
        else :
            self.S = s_min
        transform = A.Compose(
                [
                    A.Normalize(),
                    A.SmallestMaxSize(max_size=self.S),
                    A.RandomCrop(height =224,width=224),
                    A.HorizontalFlip(),
                    # A.RGBShift()
                ]

            )
        super().__init__(root,transform=transform)
        self.val =val
        self.multi = multi
    def __getitem__(self, index: int) :
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img = Image.open(
            os.path.join(
                self.root,
                "256_ObjectCategories",
                self.categories[self.y[index]],
                f"{self.y[index] + 1:03d}_{self.index[index]:04d}.jpg",
            )
        )
        # if img.shape[]
        # print(img.mode)
        if img.mode == 'L' : img = img.convert('RGB')
        ### original  height,width,channel  -> channel,height,width
        ### target list 변환시  transpose를 사용하면 오류가 발생한다.
        ### 먼저 주석처리 하고 하자
        # img=np.array(img,dtype=np.float32).transpose((2,0,1))
        img=np.array(img,dtype=np.float32)

        target = self.y[index]
        if self.transform is not None:
            img = self.transform(image=img)
            if len(img['image'].shape) == 3 and self.val==False :
                img = A.RGBShift()(image=img['image'])
            img = img['image']

        if self.target_transform is not None:
            target = self.target_transform(target)
        # print(img)
        img=img.transpose((2,0,1))
        return img, target
    
    
    
class Custom_Cifar(CIFAR100) :
    def __init__(self,root,transform = None,multi=False,s_max=None,s_min=256,download=False,val=False,train=True):

        self.multi = multi
        self.s_max = 512
        self.s_min= 256
        if multi :
            self.S = np.random.randint(low=self.s_min,high=self.s_max)
        else :
            self.S = s_min
            transform = A.Compose(
                    [
                        A.Normalize(mean =(0.5071, 0.4867, 0.4408) , std = (0.2675, 0.2565, 0.2761)),
                        A.SmallestMaxSize(max_size=self.S),
                        A.RandomCrop(height =224,width=224),
                        A.HorizontalFlip(),
                        # A.RGBShift()
                    ]

            )
        super().__init__(root,transform=transform,train=train,download=download)
        self.val =train
        self.multi = multi
    def __getitem__(self, index: int) :
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img)

        if img.mode == 'L' : img = img.convert('RGB')
        img=np.array(img,dtype=np.float32)
        
        
        if self.transform is not None:
            img = self.transform(image=img)
            if len(img['image'].shape) == 3 and self.val==False :
                img = A.RGBShift()(image=img['image'])
            img = img['image']

        if self.target_transform is not None:
            target = self.target_transform(target)
        # print(img)
        img=img.transpose((2,0,1))

        return img, target
    


class Custom_Cifar_10(CIFAR10) :
    def __init__(self,root,transform = None,multi=False,s_max=None,s_min=256,download=False,val=False,train=True):

        self.multi = multi
        self.s_max = 512
        self.s_min= 256
        if multi :
            self.S = np.random.randint(low=self.s_min,high=self.s_max)
        else :
            self.S = s_min
            transform = A.Compose(
                    [
                        A.Normalize(mean =(0.5071, 0.4867, 0.4408) , std = (0.2675, 0.2565, 0.2761)),
                        A.SmallestMaxSize(max_size=self.S),
                        A.RandomCrop(height =224,width=224),
                        A.HorizontalFlip(),
                        # A.RGBShift()
                    ]

            )
        super().__init__(root,transform=transform,train=train,download=download)
        self.val =train
        self.multi = multi
    def __getitem__(self, index: int) :
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img)

        if img.mode == 'L' : img = img.convert('RGB')
        img=np.array(img,dtype=np.float32)
        
        
        if self.transform is not None:
            img = self.transform(image=img)
            if len(img['image'].shape) == 3 and self.val==False :
                img = A.RGBShift()(image=img['image'])
            img = img['image']

        if self.target_transform is not None:
            target = self.target_transform(target)
        # print(img)
        img=img.transpose((2,0,1))

        return img, target
    
    
class Cusotm_MNIST(MNIST) :
    def __init__(self,root,transform = None,multi=False,s_max=None,s_min=256,download=False,val=False,train=True):

        self.multi = multi
        self.s_max = 512
        self.s_min= 256
        if multi :
            self.S = np.random.randint(low=self.s_min,high=self.s_max)
        else :
            self.S = s_min
            transform = A.Compose(
                    [
                        A.Normalize(),
                        A.SmallestMaxSize(max_size=self.S),
                        A.RandomCrop(height =224,width=224),
                        A.HorizontalFlip(),
                        # A.RGBShift()
                    ]

            )
        super().__init__(root,transform=transform,train=train,download=download)
        self.val =train
        self.multi = multi
    def __getitem__(self, index: int) :
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img=np.array(img)
        img = Image.fromarray(img)

        if img.mode == 'L' : img = img.convert('RGB')
        img=np.array(img,dtype=np.float32)
        
        
        if self.transform is not None:
            img = self.transform(image=img)
            if len(img['image'].shape) == 3 and self.val==False :
                img = A.RGBShift()(image=img['image'])
            img = img['image']

        if self.target_transform is not None:
            target = self.target_transform(target)
        # print(img)
        img=img.transpose((2,0,1))

        return img, target
    
class Cusotm_ImageNet(ImageNet) :
    def __init__(self,root,transform = None,multi=False,s_max=None,s_min=256,split=None,val=False):

        self.multi = multi
        self.s_max = 512
        self.s_min= 256
        if multi :
            self.S = np.random.randint(low=self.s_min,high=self.s_max)
        else :
            self.S = s_min
            transform = A.Compose(
                    [
                        A.Normalize(),
                        A.SmallestMaxSize(max_size=self.S),
                        A.RandomCrop(height =224,width=224),
                        A.HorizontalFlip(),
                        # A.RGBShift()
                    ]

            )
        super().__init__(root,transform=transform,split=split)
        self.val =val
        self.multi = multi
    def __getitem__(self, index: int) :
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img=np.array(img)
        img = Image.fromarray(img)

        if img.mode == 'L' : img = img.convert('RGB')
        img=np.array(img,dtype=np.float32)
        
        # print(self.transform)
        
        if self.transform is not None:
            img = self.transform(image=img)
            if len(img['image'].shape) == 3 and self.val==False :
                img = A.RGBShift()(image=img['image'])
            img = img['image']

        if self.target_transform is not None:
            target = self.target_transform(target)
        # print(img)
        img=img.transpose((2,0,1))

        return img, target
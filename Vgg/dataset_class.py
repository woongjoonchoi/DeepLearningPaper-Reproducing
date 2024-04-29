import albumentations as A
import numpy as np


from torchvision.datasets import Caltech256 ,Caltech101 ,CIFAR100
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
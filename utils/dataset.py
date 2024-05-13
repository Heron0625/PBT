import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import torch
import os.path as osp
import sys
from torch.utils.data.dataset import Dataset
import random
import cv2
from albumentations import (
    RandomRotate90, Transpose, ShiftScaleRotate, Blur,
    OpticalDistortion, CLAHE, GaussNoise, MotionBlur,
    GridDistortion, HueSaturationValue,ToGray,
    MedianBlur, PiecewiseAffine, Sharpen, Emboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
import numpy as np
def strong_aug(p=0.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        ToGray(),
        OneOf([
            GaussNoise(),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            PiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            Sharpen(),
            Emboss(),
            RandomBrightnessContrast()
        ], p=0.5),
        HueSaturationValue(p=0.5),
    ], p=p)

class TrainSetLoader(Dataset):
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id ,base_size=512,crop_size=480,transform=None,suffix='.png', aug = 0., useprior=True):
        super(TrainSetLoader, self).__init__()
        self.useprior = useprior
        self.transform = transform
        self._items = img_id
        self.masks = dataset_dir+'/'+'masks'
        self.images = dataset_dir+'/'+'images'
        self.prior = dataset_dir+'/'+'masks'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix
        self.aug = aug
    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BICUBIC)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img_1 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        mask_1 = cv2.cvtColor(np.asarray(mask), cv2.COLOR_RGB2BGR)


        data = {"image": img_1, "mask": mask_1,}
        augmentation = strong_aug(p=self.aug)
        augmented = augmentation(**data)  ## 数据增强

        img_1, mask_1 = augmented["image"], augmented["mask"]
        img = Image.fromarray(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
        mask = Image.fromarray(cv2.cvtColor(mask_1, cv2.COLOR_BGR2GRAY))

        return img, mask,



    def __getitem__(self, idx):
        cv2.setNumThreads(0)

        img_id     = self._items[idx]                        # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path   = self.images+'/'+img_id+self.suffix   # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks +'/'+img_id+self.suffix
        prior_path = self.prior +'/'+img_id+self.suffix

        img = Image.open(img_path).convert('RGB')         ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        mask = Image.open(label_path).convert('L')
        if self.useprior == True:
            prior = Image.open(prior_path).convert('L')
            # prior = mask
        else:
            prior = None
        # synchronized transform
        img, mask, prior = self._sync_transform(img=img, mask=mask, prior=prior)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32')/ 255.0
        if prior is not None:
            prior = np.expand_dims(prior, axis=0).astype('float32') / 255.0
            prior = torch.from_numpy(prior)
        else:
            prior = torch.zeros(1)
        return img, torch.from_numpy(mask), prior, #img_id[-1]

    def __len__(self):
        return len(self._items)


class TestSetLoader(Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id,transform=None,base_size=512,crop_size=480,suffix='.png'):
        super(TestSetLoader, self).__init__()
        self.transform = transform
        self._items    = img_id
        self.masks     = dataset_dir+'/'+'masks'
        self.images    = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BICUBIC)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        # final transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)  # img: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        return img, mask

    def __getitem__(self, idx):
        # print('idx:',idx)
        cv2.setNumThreads(0)

        img_id = self._items[idx]  # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path   = self.images+'/'+img_id+self.suffix    # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks +'/'+img_id+self.suffix
        img  = Image.open(img_path).convert('RGB')  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        mask = Image.open(label_path)
        # synchronized transform
        img, mask = self._testval_sync_transform(img, mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0


        return img, torch.from_numpy(mask)  # img_id[-1]

    def __len__(self):
        return len(self._items)

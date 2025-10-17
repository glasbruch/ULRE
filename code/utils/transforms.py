""" From: https://github.com/vojirt/PixOOD/blob/main/code/dataloaders/transforms.py
"""
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import ToTensor, Normalize, InterpolationMode, PILToTensor, RandomCrop

class ToTensorSN():
    def __init__(self, normalize01=True):
        if normalize01:
            self._op = ToTensor()
        else:
            self._op = PILToTensor()
        self._op_target = PILToTensor()
    def __call__(self, image, label, *inputs, **kwargs):
        image = self._op(image)
        if label is not None:
            label = self._op_target(label).long().squeeze()
        return image, label, tuple(i for i in inputs)

class NormalizeSN():
    def __init__(self, mean, std):
        self._op = Normalize(mean=mean, std=std)

    def __call__(self, image, label, *inputs, **kwargs):
        image = self._op(image)
        return image, label, tuple(i for i in inputs)

class ResizeLongestSideDivisible():
    def __init__(self, img_size, divider, eval_mode=False, randomcrop=False, hflip=False):
        self.divider = divider
        self.eval_mode = eval_mode
        self.randomcrop = randomcrop
        self.hflip = hflip

        # Transform input to DINO format:
        #  - can have arbitrary size, but it must be divisible by 14 (patch size of the used VIT backbone)
        #  - keeps aspect ratio, no padding needed
        if isinstance(img_size, list) and len(img_size) == 2:
            self.img_sz = img_size
            if self.img_sz[0] % self.divider > 0 or self.img_sz[1] % self.divider > 0:
                raise RuntimeError(f"INPUT.IMG_SIZE has to be divisible by 14")
        elif isinstance(img_size, int) and img_size % self.divider == 0:
            # longest side stored in IMG_SIZE
            self.img_sz = img_size
        else:
            raise RuntimeError(f"INPUT.IMG_SIZE has to be list[2] or int and divisible by {divider}!")

    def __call__(self, image, label, *inputs, **kwargs):
        x_size = image.shape[-2:]

        if not self.eval_mode:
            if self.randomcrop:
                # i, j, h, w = RandomResizedCrop.get_params(x_sn.image, scale=[0.75, 1.0], ratio=[0.75, 4.0/3.0])

                if x_size[0] >= x_size[1]:
                    factor = x_size[0] / float(self.img_sz)
                    size = [int(self.img_sz), int(self.divider*((x_size[1] / factor) // self.divider))] 
                else:
                    factor = x_size[1] / float(self.img_sz)
                    size = [int(self.divider*((x_size[0] / factor) // self.divider)), int(self.img_sz)] 

                i, j, h, w = RandomCrop.get_params(image, size)
                image = F.crop(image, i, j, h, w)
                label = F.crop(label, i, j, h, w)

            if self.hflip and torch.rand(1) < 0.5:
                image = F.hflip(image)
                label = F.hflip(label)
        else:
            # assuming x is tensor of [..., h, w] shape
            self.img_sz = int((np.max(x_size) // self.divider) * self.divider)

        if isinstance(self.img_sz, list):
            size = self.img_sz
        else:
            if x_size[0] >= x_size[1]:
                factor = x_size[0] / float(self.img_sz)
                size = [int(self.img_sz), int(self.divider*((x_size[1] / factor) // self.divider))] 
            else:
                factor = x_size[1] / float(self.img_sz)
                size = [int(self.divider*((x_size[0] / factor) // self.divider)), int(self.img_sz)] 
        image = torchvision.transforms.functional.resize(image, size, antialias=True)
        if label is not None:
            label = torchvision.transforms.functional.resize(label[None, ...], size, interpolation=InterpolationMode.NEAREST)[0, ...]
        return image, label, tuple(i for i in inputs)

"""
class ResizeLongestSideDivisible():
    def __init__(self, img_size, divider, eval_mode=False, randomcrop=False, hflip=False):
        self.divider = divider
        self.eval_mode = eval_mode
        self.randomcrop = randomcrop
        self.hflip = hflip

        # Transform input to DINO format:
        #  - can have arbitrary size, but it must be divisible by 14 (patch size of the used VIT backbone)
        #  - keeps aspect ratio, no padding needed
        if isinstance(img_size, list) and len(img_size) == 2:
            self.img_sz = img_size
            if self.img_sz[0] % self.divider > 0 or self.img_sz[1] % self.divider > 0:
                raise RuntimeError(f"INPUT.IMG_SIZE has to be divisible by 14")
        elif isinstance(img_size, int) and img_size % self.divider == 0:
            # longest side stored in IMG_SIZE
            self.img_sz = img_size
        else:
            raise RuntimeError(f"INPUT.IMG_SIZE has to be list[2] or int and divisible by {divider}!")

    def __call__(self, inputs):
        image = inputs["image"]
        label = inputs["mask"]

        x_size = image.shape[-2:]

        if not self.eval_mode:
            if self.randomcrop:
                # i, j, h, w = RandomResizedCrop.get_params(x_sn.image, scale=[0.75, 1.0], ratio=[0.75, 4.0/3.0])

                if x_size[0] >= x_size[1]:
                    factor = x_size[0] / float(self.img_sz)
                    size = [int(self.img_sz), int(self.divider*((x_size[1] / factor) // self.divider))] 
                else:
                    factor = x_size[1] / float(self.img_sz)
                    size = [int(self.divider*((x_size[0] / factor) // self.divider)), int(self.img_sz)] 

                i, j, h, w = RandomCrop.get_params(image, size)
                image = F.crop(image, i, j, h, w)
                label = F.crop(label, i, j, h, w)

            if self.hflip and torch.rand(1) < 0.5:
                image = F.hflip(image)
                label = F.hflip(label)
        else:
            # assuming x is tensor of [..., h, w] shape
            self.img_sz = int((np.max(x_size) // self.divider) * self.divider)

        if isinstance(self.img_sz, list):
            size = self.img_sz
        else:
            if x_size[0] >= x_size[1]:
                factor = x_size[0] / float(self.img_sz)
                size = [int(self.img_sz), int(self.divider*((x_size[1] / factor) // self.divider))] 
            else:
                factor = x_size[1] / float(self.img_sz)
                size = [int(self.divider*((x_size[0] / factor) // self.divider)), int(self.img_sz)] 
        image = torchvision.transforms.functional.resize(image, size, antialias=True)
        if label is not None:
            label = torchvision.transforms.functional.resize(label[None, ...], size, interpolation=InterpolationMode.NEAREST)[0, ...]

        inputs["image"] = image
        inputs["mask"] = label
        return inputs"""
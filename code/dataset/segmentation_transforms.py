import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
import numpy as np

import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

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

    def __call__(self, sample, *inputs, **kwargs):
        image, label = sample['image'], sample['mask']
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
        image = transforms.functional.resize(image, size, antialias=True)
        if label is not None:
            label = transforms.functional.resize(label[None, ...], size, interpolation=InterpolationMode.NEAREST)[0, ...]
        return {'image': image, 'mask': label}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(np.array(mask)).long()
        
        return {'image': image, 'mask': mask}
        # Mask should have shape B x H x W -> squeeze channel dimension.
        #return {'image': image, 'mask': mask.squeeze().long()}

class Resize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = TF.resize(image, self.output_size, interpolation=transforms.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.output_size, interpolation=transforms.InterpolationMode.NEAREST)
        return {'image': image, 'mask': mask}
    
class RandomHorizontalFlip(object):
    """Horizontally flip the given image and mask randomly with a given probability."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        return {'image': image, 'mask': mask}
    
class RandomCrop:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=self.output_size)
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        return {'image': image, 'mask': mask}

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = TF.normalize(image, self.mean, self.std)
        return {'image': image, 'mask': mask}
    
class PaddingTransform:
    def __init__(self, padding, fill=0, padding_mode='constant'):
        """
        Initializes the PaddingTransform class.
        
        Args:
            padding (int or tuple): The size of the padding. If a single int is provided, the same padding is applied to all sides. If a tuple of four values is provided, they correspond to padding for left, top, right, and bottom respectively.
            fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple is used, the values correspond to each channel.
            padding_mode (str): Type of padding. Can be 'constant', 'edge', 'reflect', or 'symmetric'.
        """
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, sample):
        """
        Apply padding to both image and mask.
        
        Args:
            image (PIL.Image): The image to be padded.
            mask (PIL.Image): The mask to be padded.

        Returns:
            PIL.Image: Padded image and mask.
        """
        image, mask = sample['image'], sample['mask']
        image = TF.pad(image, self.padding, fill=self.fill, padding_mode=self.padding_mode)
        mask = TF.pad(mask, self.padding, fill=0, padding_mode=self.padding_mode)  # Typically padding masks with 0
        return {'image': image, 'mask': mask}
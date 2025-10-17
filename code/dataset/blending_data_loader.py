import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataset.segmentation_transforms import RandomCrop, ToTensor, Normalize, RandomHorizontalFlip, ResizeLongestSideDivisible
from torchvision.transforms import v2
import torchvision.tv_tensors as tv_tensors

from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageTransform
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2
from dataset.training.cityscapes import Cityscapes
from dataset.training.coco import COCO

from utils.img_utils import generate_random_crop_pos, random_crop_pad_to_shape

def pil_to_cv2(pil_image):
    """
    Convert a PIL Image to a CV2 Image (numpy array).
    Handles both RGB and grayscale images.
    
    Args:
        pil_image (PIL.Image.Image): PIL Image object
        
    Returns:
        numpy.ndarray: OpenCV image (BGR format for color images, grayscale for grayscale images)
    
    Note: OpenCV uses BGR format while PIL uses RGB format for color images
    """
    # Get the image mode
    mode = pil_image.mode
    
    # Convert to numpy array
    numpy_image = np.array(pil_image)
    
    if mode == 'L':  # Grayscale
        # No need for color space conversion for grayscale
        return numpy_image
    elif mode == 'RGB':
        # Convert RGB to BGR format for OpenCV
        return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    elif mode == 'RGBA':
        # Convert RGBA to RGB first, then to BGR
        rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
        rgb_image.paste(pil_image, mask=pil_image.split()[3])
        return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
    else:
        # For other modes, convert to RGB first
        rgb_image = pil_image.convert('RGB')
        return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """
    Convert a CV2 Image (numpy array) to a PIL Image.
    Handles both BGR and grayscale images.
    
    Args:
        cv2_image (numpy.ndarray): OpenCV image (BGR format for color images, 
                                  grayscale for grayscale images)
        
    Returns:
        PIL.Image.Image: PIL Image object (RGB format for color images, 
                        L mode for grayscale images)
    
    Note: OpenCV uses BGR format while PIL uses RGB format for color images
    """
    # Check if image is grayscale
    if len(cv2_image.shape) == 2 or (len(cv2_image.shape) == 3 and cv2_image.shape[2] == 1):
        # Convert single channel grayscale to PIL
        return Image.fromarray(cv2_image, mode='L')
    else:
        # Convert BGR to RGB format
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image, mode='RGB')

'''
class MixingDataset(Dataset):
    def __init__(self, 
                 cityscape_root, 
                 coco_root, 
                 split='train', 
                 img_size=(512, 1024),
                 adjust_brightness=True,
                 color_transfer=True):
        self.cityscape_root = cityscape_root
        self.pascal_voc_root = coco_root
        self.split = split
        self.img_size = img_size

        self.adjust_brightness = adjust_brightness
        self.color_transfer = color_transfer

        self.cs = Cityscapes(root=cityscape_root, split=split)
        self.coco = COCO(root=coco_root, split=split, proxy_size=None)

        self.coco_number = len(self.coco.images)

        self.transform = transforms.Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomHorizontalFlip()
        ])
        """self.transform = v2.Compose([
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            v2.RandomHorizontalFlip(p=0.5),
            # Convert image to float32 and scale to [0,1]. Mask dtype is preserved (e.g. uint8).
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])"""
    
    def _random_scale(self, img, gt=None, scales=None):
        scale = random.choice(scales)
        # scale = random.uniform(scales[0], scales[-1])
        sw = int(img.size[0] * scale)
        sh = int(img.size[1] * scale)
        #img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
        img = img.resize((sw, sh), Image.BILINEAR)
        if gt is not None:
            #gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
            gt = gt.resize((sw, sh), Image.NEAREST)

        return img, gt, scale
    
    def _crop_random_section(self, image, mask, crop_width, crop_height):
        # Open the image
        #image = Image.open(image_path)
        img_width, img_height = image.size
    
        # Ensure crop size is smaller than the image size
        if crop_width > img_width or crop_height > img_height:
            raise ValueError("Crop size is larger than the image dimensions.")
    
        # Generate random coordinates for the top-left corner of the crop
        left = random.randint(0, img_width - crop_width)
        top = random.randint(0, img_height - crop_height)
        right = left + crop_width
        bottom = top + crop_height
    
        # Crop the image
        cropped_image = image.crop((left, top, right, bottom))
        cropped_mask = mask.crop((left, top, right, bottom))
    
        return cropped_image, cropped_mask
    
    def _adjust_brightness_contrast(self, image, target_brightness, target_contrast):
        brightness_factor = target_brightness / image.convert('L').getextrema()[1]
        contrast_factor = target_contrast / (image.convert('L').getextrema()[1] - image.convert('L').getextrema()[0])
        
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        
        return image

    def _color_transfer(self, source, target):
        # Convert images to LAB color space
        source_lab = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2LAB).astype(float)
        target_lab = cv2.cvtColor(np.array(target), cv2.COLOR_RGB2LAB).astype(float)

        # Compute mean and std for each channel
        source_mean, source_std = [], []
        target_mean, target_std = [], []
        for i in range(3):
            source_mean.append(np.mean(source_lab[:,:,i]))
            source_std.append(np.std(source_lab[:,:,i]))
            target_mean.append(np.mean(target_lab[:,:,i]))
            target_std.append(np.std(target_lab[:,:,i]))

        # Adjust each channel
        for i in range(3):
            source_lab[:,:,i] = ((source_lab[:,:,i] - source_mean[i]) * (target_std[i] / (source_std[i] + 1e-8))) + target_mean[i]

        # Clip values to valid range
        source_lab = np.clip(source_lab, 0, 255)

        # Convert back to RGB
        result = cv2.cvtColor(source_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return Image.fromarray(result)

    def _extract_object(self, image, mask):
        # Convert mask to numpy array
        mask_np = np.array(mask)
        
        # Find the bounding box of the object
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Crop the image and mask to the bounding box
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        cropped_mask = mask.crop((x_min, y_min, x_max, y_max))
        
        # Convert mask properly
        """
        cropped_mask = np.array(cropped_mask)
        cropped_mask[cropped_mask==254] = 255
        cropped_mask = Image.fromarray(cropped_mask, mode="L")"""

        return cropped_image, cropped_mask
        """
        # Create an RGBA image
        rgba_image = Image.new('RGBA', cropped_image.size)
        rgba_image.paste(cropped_image, (0, 0))
        
        # Apply the mask
        data = np.array(rgba_image)
        mask_data = np.array(cropped_mask)
        data[:, :, 3] = mask_data
        
        return Image.fromarray(data)"""

    def _insert_anomaly(self, image, label):
        # Randomly select a COCO object
        #while True:
             # Some images don not contain a suitable object -> resample
        #    coco_idx = random.randint(0, self.coco_number -1 )
        #    anomaly, anomaly_mask = self.coco[coco_idx]
        #    if 254 in np.unique(anomaly_mask):
        #        break

        # Select a valid anomaly from COCO
        coco_indices = np.random.permutation(self.coco_number)
        for coco_idx in coco_indices:
            anomaly, anomaly_mask = self.coco[coco_idx]
            if 254 in np.unique(anomaly_mask):
                break

        anomaly_object, object_mask = self._extract_object(anomaly, anomaly_mask)

        # New crop
        """
        image = pil_to_cv2(image)
        label = pil_to_cv2(label)
        crop_size = (self.img_size, self.img_size)
        crop_pos = generate_random_crop_pos(image.shape[:2], crop_size)

        image, _ = random_crop_pad_to_shape(image, crop_pos, crop_size, 0)
        label, _ = random_crop_pad_to_shape(label, crop_pos, crop_size, 255)
        image = cv2_to_pil(image)
        label = cv2_to_pil(label)"""
        
        sample = {"image": image, "mask": label}
        #sample =v2.RandomCrop(size=self.img_size)(sample)
        sample = v2.RandomResizedCrop(size=self.img_size, scale=(0.5, 2.0), antialias=True)(sample)
        image = sample["image"]
        label = sample["mask"]

        # Old resize
        """
        # Resize anomaly to a random size
        original_size = anomaly_object.size
        # Original config
        anomaly_size = random.randint(100, min(image.size) // 2)
        # Dino_log-loss_hidden-dim512_noise-std0_hidden_states21_lr2e-5_bias_min-size150_max-size3-4
        #anomaly_size = random.randint(150, (min(image.size) * 3 )// 4)
        # Dino_log-loss_hidden-dim512_noise-std0_hidden_states21_lr2e-5_bias_min-size150_max-size
        #anomaly_size = random.randint(150, min(image.size))
        scale_factor = anomaly_size / max(original_size)
        new_size = tuple(int(dim * scale_factor) for dim in original_size)
        anomaly_object = anomaly_object.resize(new_size, Image.LANCZOS)
        object_mask = object_mask.resize(new_size, Image.LANCZOS)"""

        ####### Fixed size for debugging
        # Resize anomaly to a random size
        original_size = anomaly_object.size
        # Original config
        anomaly_size = int(min(image.size) * 3.0/4.0)
        scale_factor = anomaly_size / max(original_size)
        new_size = tuple(int(dim * scale_factor) for dim in original_size)
        anomaly_object = anomaly_object.resize(new_size, Image.LANCZOS)
        object_mask = object_mask.resize(new_size, Image.LANCZOS)

        # Random position for insertion
        x = random.randint(0, image.size[0] - new_size[0])
        y = random.randint(0, image.size[1] - new_size[1])
        
        # Extract the region where the anomaly will be inserted
        target_region = image.crop((x, y, x + new_size[0], y + new_size[1]))
        
        # Adjust brightness and contrast
        if self.adjust_brightness:
            target_brightness = target_region.convert('L').getextrema()[1]
            target_contrast = target_region.convert('L').getextrema()[1] - target_region.convert('L').getextrema()[0]
            anomaly_object = self._adjust_brightness_contrast(anomaly_object, target_brightness, target_contrast)
        
        # Color transfer
        if self.color_transfer:
            anomaly_object = self._color_transfer(anomaly_object, target_region)
    
        # Create an RGBA image
        rgba_image = Image.new('RGBA', anomaly_object.size)
        rgba_image.paste(anomaly_object, (0, 0))
        
        # Insert mask in the alpha channel
        data = np.array(rgba_image)
        mask_data = np.array(object_mask)

        data[:, :, 3] = mask_data
        anomaly_object = Image.fromarray(data)

        # Insert anomaly
        image.paste(anomaly_object, (x, y), anomaly_object.split()[3])
        
        # Update label
        label = np.array(label)
        object_mask = np.array(anomaly_object.split()[3])
        label[y:y+new_size[1], x:x+new_size[0]][object_mask != 0] = 254  # Assign anomaly label (254)
        
        return image, Image.fromarray(label)

    def __getitem__(self, index):
        # Load Cityscapes image and label
        image, label = self.cs[index]

        # Insert anomaly with 50% probability
        #if random.random() < 0.5:
        image, label = self._insert_anomaly(image, label)
        
        # Normalization should only apply to the image.
        #image = tv_tensors.Image(image)
        #label = tv_tensors.Mask(label, dtype=torch.long) # Mask should be uint8 or long

        sample = {'image': image, 'mask': label}
        # Apply transformations
        sample = self.transform(sample)
        image = sample["image"]
        label = sample["mask"]

        label = torch.from_numpy(np.array(label)).long()

        #label = sample["mask"].data # Access the underlying tensor data
        #label = label.squeeze(0).to(torch.long) # Squeeze channel & convert to long

        return image, label

    def __len__(self):
        return len(self.cs)'''

class MixingDataset(Dataset):
    def __init__(self, 
                 cityscape_root, 
                 coco_root, 
                 split='train', 
                 img_size=(512, 1024),
                 adjust_brightness=True,
                 color_transfer=True):
        self.cityscape_root = cityscape_root
        self.pascal_voc_root = coco_root
        self.split = split
        self.img_size = img_size

        self.adjust_brightness = adjust_brightness
        self.color_transfer = color_transfer

        self.cs = Cityscapes(root=cityscape_root, split=split)
        self.coco = COCO(root=coco_root, split=split, proxy_size=None)

        self.coco_number = len(self.coco.images)

        self.transform = transforms.Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomHorizontalFlip()
        ])
    
    def _random_scale(self, img, gt=None, scales=None):
        scale = random.choice(scales)
        # scale = random.uniform(scales[0], scales[-1])
        sw = int(img.size[0] * scale)
        sh = int(img.size[1] * scale)
        #img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
        img = img.resize((sw, sh), Image.BILINEAR)
        if gt is not None:
            #gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
            gt = gt.resize((sw, sh), Image.NEAREST)

        return img, gt, scale
    
    def _crop_random_section(self, image, mask, crop_width, crop_height):
        # Open the image
        #image = Image.open(image_path)
        img_width, img_height = image.size
    
        # Ensure crop size is smaller than the image size
        if crop_width > img_width or crop_height > img_height:
            raise ValueError("Crop size is larger than the image dimensions.")
    
        # Generate random coordinates for the top-left corner of the crop
        left = random.randint(0, img_width - crop_width)
        top = random.randint(0, img_height - crop_height)
        right = left + crop_width
        bottom = top + crop_height
    
        # Crop the image
        cropped_image = image.crop((left, top, right, bottom))
        cropped_mask = mask.crop((left, top, right, bottom))
    
        return cropped_image, cropped_mask
    
    def _adjust_brightness_contrast(self, image, target_brightness, target_contrast):
        brightness_factor = target_brightness / image.convert('L').getextrema()[1]
        contrast_factor = target_contrast / (image.convert('L').getextrema()[1] - image.convert('L').getextrema()[0])
        
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        
        return image

    def _color_transfer(self, source, target):
        # Convert images to LAB color space
        source_lab = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2LAB).astype(float)
        target_lab = cv2.cvtColor(np.array(target), cv2.COLOR_RGB2LAB).astype(float)

        # Compute mean and std for each channel
        source_mean, source_std = [], []
        target_mean, target_std = [], []
        for i in range(3):
            source_mean.append(np.mean(source_lab[:,:,i]))
            source_std.append(np.std(source_lab[:,:,i]))
            target_mean.append(np.mean(target_lab[:,:,i]))
            target_std.append(np.std(target_lab[:,:,i]))

        # Adjust each channel
        for i in range(3):
            source_lab[:,:,i] = ((source_lab[:,:,i] - source_mean[i]) * (target_std[i] / (source_std[i] + 1e-8))) + target_mean[i]

        # Clip values to valid range
        source_lab = np.clip(source_lab, 0, 255)

        # Convert back to RGB
        result = cv2.cvtColor(source_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return Image.fromarray(result)

    def _extract_object(self, image, mask):
        # Convert mask to numpy array
        mask_np = np.array(mask)
        
        # Find the bounding box of the object
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Crop the image and mask to the bounding box
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        cropped_mask = mask.crop((x_min, y_min, x_max, y_max))

        return cropped_image, cropped_mask

    def _insert_anomaly(self, image, label):
        # Randomly select a COCO object
        # Select a valid anomaly from COCO
        coco_indices = np.random.permutation(self.coco_number)
        for coco_idx in coco_indices:
            anomaly, anomaly_mask = self.coco[coco_idx]
            if 254 in np.unique(anomaly_mask):
                break

        anomaly_object, object_mask = self._extract_object(anomaly, anomaly_mask)

        sample = {"image": image, "mask": label}
        sample =v2.RandomCrop(size=self.img_size)(sample)
        image = sample["image"]
        label = sample["mask"]

        # Resize anomaly to a random size
        original_size = anomaly_object.size
        # Original config
        anomaly_size = random.randint(100, min(image.size) // 2)
        scale_factor = anomaly_size / max(original_size)
        new_size = tuple(int(dim * scale_factor) for dim in original_size)
        anomaly_object = anomaly_object.resize(new_size, Image.LANCZOS)
        object_mask = object_mask.resize(new_size, Image.LANCZOS)

        # Random position for insertion
        x = random.randint(0, image.size[0] - new_size[0])
        y = random.randint(0, image.size[1] - new_size[1])
        
        # Extract the region where the anomaly will be inserted
        target_region = image.crop((x, y, x + new_size[0], y + new_size[1]))
        
        # Adjust brightness and contrast
        if self.adjust_brightness:
            target_brightness = target_region.convert('L').getextrema()[1]
            target_contrast = target_region.convert('L').getextrema()[1] - target_region.convert('L').getextrema()[0]
            anomaly_object = self._adjust_brightness_contrast(anomaly_object, target_brightness, target_contrast)
        
        # Color transfer
        if self.color_transfer:
            anomaly_object = self._color_transfer(anomaly_object, target_region)
    
        # Create an RGBA image
        rgba_image = Image.new('RGBA', anomaly_object.size)
        rgba_image.paste(anomaly_object, (0, 0))
        
        # Insert mask in the alpha channel
        data = np.array(rgba_image)
        mask_data = np.array(object_mask)

        data[:, :, 3] = mask_data
        anomaly_object = Image.fromarray(data)

        # Insert anomaly
        image.paste(anomaly_object, (x, y), anomaly_object.split()[3])
        
        # Update label
        label = np.array(label)
        object_mask = np.array(anomaly_object.split()[3])
        label[y:y+new_size[1], x:x+new_size[0]][object_mask != 0] = 254  # Assign anomaly label (254)
        
        return image, Image.fromarray(label)

    def __getitem__(self, index):
        # Load Cityscapes image and label
        image, label = self.cs[index]

        image, label = self._insert_anomaly(image, label)
        
        sample = {'image': image, 'mask': label}
        # Apply transformations
        sample = self.transform(sample)
        image = sample["image"]
        label = sample["mask"]

        label = torch.from_numpy(np.array(label)).long()

        return image, label

    def __len__(self):
        return len(self.cs)

def get_mix_loader(cityscape_root, 
                                coco_root, 
                                img_size=900, 
                                batch_size=8, 
                                num_workers=4, 
                                split='train', 
                                adjust_brightness=True, 
                                color_transfer=True,
                                ):
    
    dataset = MixingDataset(cityscape_root, 
                                     coco_root, 
                                     split, 
                                     img_size, 
                                     adjust_brightness, 
                                     color_transfer)
    return DataLoader(dataset, 
                      batch_size=batch_size, 
                      shuffle=(split == 'train'), 
                      num_workers=num_workers,
                      pin_memory=True)

if __name__ == "__main__":
    loader = get_mix_loader(cityscape_root='path/to/your/city_scape',
                                         coco_root='path/to/your/coco',
                                         adjust_brightness=True, color_transfer=True)
    def norm_img(img):
        n_channels, h, w = img.shape
        img = img.view(img.size(0), -1)
        img -= img.min(1, keepdim=True)[0]
        img /= img.max(1, keepdim=True)[0]
        img = img.view(n_channels, h, w)
        return img

    for image, label in loader:
        #print(image.shape)
        print(np.unique(label))
        fig, axis = plt.subplots(8,2,figsize=(8,16))
        for idx in range(8):
            img = norm_img(image[idx])
            axis[idx,0].imshow(img.permute(1,2,0))
            axis[idx,0].set_axis_off()
            
            ood_object_mask = torch.zeros_like(label[idx], dtype=torch.float)
            ood_object_mask[torch.where(label[idx] == 254)] = 1
            axis[idx,1].imshow(ood_object_mask)
            axis[idx,1].set_axis_off()
        
        plt.tight_layout()
        plt.savefig("loader_test_pad_crop.png")
        break


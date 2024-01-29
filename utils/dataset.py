from torch.utils.data import Dataset
import os
from PIL import Image
import torch


class PairedDataset(Dataset):
    def __init__(self, root_dir_low, root_dir_high, transform=None):
        self.root_dir_low = root_dir_low
        self.root_dir_high = root_dir_high
        self.transform = transform
        self.low_list = [os.path.join(self.root_dir_low, img) for img in os.listdir(
            self.root_dir_low) if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.bmp') or img.endswith('.JPG')]
        self.high_list = [os.path.join(self.root_dir_high, img) for img in os.listdir(
            self.root_dir_high) if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.bmp') or img.endswith('.JPG')]
        self.low_imgs = sorted(self.low_list)
        self.high_imgs = sorted(self.high_list)
    
    def __len__(self):
        return len(self.low_imgs)
    
    def __getitem__(self, idx):
        low_img = self.low_imgs[idx]
        low_img = Image.open(low_img)
        high_img = self.high_imgs[idx]
        high_img = Image.open(high_img)
        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)
        
        paired_img = torch.cat((low_img, high_img), dim=0)
        
        return paired_img, self.low_imgs, self.high_imgs
    

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_list = [os.path.join(self.root_dir, img) for img in os.listdir(
            self.root_dir) if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.bmp') or img.endswith('.JPG')]
        self.imgs = sorted(self.img_list)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = Image.open(img)
        if self.transform:
            img = self.transform(img)
        
        return img, self.imgs
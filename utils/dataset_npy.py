from torch.utils.data import Dataset
import os
import numpy as np
import torch
import cv2


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = sorted([os.path.join(self.root_dir, img) for img in os.listdir(
            self.root_dir) if img.endswith('.npy')])
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = np.load(img)
        if self.transform:
            # img = self.transform(img)
            img = cv2.resize(img, (512, 512))
        
        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1)
        return img

class PairedDataset(Dataset):
    def __init__(self, root_dir_low, root_dir_high, transform=None):
        self.root_dir_low = root_dir_low
        self.root_dir_high = root_dir_high
        self.transform = transform
        self.low_list = [os.path.join(self.root_dir_low, img) for img in os.listdir(
            self.root_dir_low) if img.endswith('.npy')]
        self.high_list = [os.path.join(self.root_dir_high, img) for img in os.listdir(
            self.root_dir_high) if img.endswith('.npy')]
        self.low_imgs = sorted(self.low_list)
        self.high_imgs = sorted(self.high_list)
    
    def __len__(self):
        return len(self.low_imgs)
    
    def __getitem__(self, idx):
        low_img = self.low_imgs[idx]
        low_img = np.load(low_img)
        high_img = self.high_imgs[idx]
        high_img = np.load(high_img)
        
        if self.transform:
            # low_img = self.transform(low_img)
            # high_img = self.transform(high_img)
            low_img = cv2.resize(low_img, (512, 512))
            high_img = cv2.resize(high_img, (512, 512))
        
        low_img = torch.tensor(low_img, dtype=torch.float32)
        high_img = torch.tensor(high_img, dtype=torch.float32)
        low_img = low_img.permute(2, 0, 1)
        high_img = high_img.permute(2, 0, 1)
        
        paired_img = torch.cat((low_img, high_img), dim=0)
        
        return paired_img, self.low_imgs, self.high_imgs
    

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_list = [os.path.join(self.root_dir, img) for img in os.listdir(
            self.root_dir) if img.endswith('.npy')]
        self.imgs = sorted(self.img_list)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = np.load(img)
        if self.transform:
            # img = self.transform(img)
            img = cv2.resize(img, (512, 512))
        
        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1)
        
        return img, self.imgs
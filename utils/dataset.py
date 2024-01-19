from torch.utils.data import Dataset
import os
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = sorted([os.path.join(self.root_dir, img) for img in os.listdir(
            self.root_dir) if img.endswith('.jpg') or img.endswith('.png')])
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = Image.open(img)
        if self.transform:
            img = self.transform(img)
        return img
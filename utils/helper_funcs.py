import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
import torch


def sp_ch_3(x):
    return x[:, :3, :, :]

def sp_ch_1(x):
    return x[:, 3:, :, :]


def PairRandomCrop(img1, img2, img3, img4, img5, img6, crop_size):
    _, _, H, W = img1.shape
    top = torch.randint(0, H - crop_size, (1,)).item()
    left = torch.randint(0, W - crop_size, (1,)).item()
    crop1 = TF.crop(img1, top, left, crop_size, crop_size)
    crop2 = TF.crop(img2, top, left, crop_size, crop_size)
    crop3 = TF.crop(img3, top, left, crop_size, crop_size)
    crop4 = TF.crop(img4, top, left, crop_size, crop_size)
    crop5 = TF.crop(img5, top, left, crop_size, crop_size)
    crop6 = TF.crop(img6, top, left, crop_size, crop_size)

    return crop1, crop2, crop3, crop4, crop5, crop6

def expand_channel(x):
    return torch.cat((x, x, x), dim=1)

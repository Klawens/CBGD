import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
import torch

def extract_illumination_map_batch(batch_images, sigma=30):
    """
    Extract the illumination map for a batch of images.

    Args:
    - batch_images (numpy.ndarray): Batch of images of shape (batch, 3, h, w)
    - sigma (float): Standard deviation for Gaussian blur

    Returns:
    - numpy.ndarray: Batch of illumination maps of shape (batch, 1, h, w)
    """
    batch_size, _, h, w = batch_images.shape
    illumination_maps = np.zeros((batch_size, 1, h, w))

    for i in range(batch_size):
        # Convert to grayscale
        gray = cv2.cvtColor(batch_images[i].transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)

        # Store the smoothed image as the illumination map
        illumination_maps[i, 0, :, :] = gray

    return illumination_maps

def sp_ch_3(x):
    return x[:, :3, :, :]

def sp_ch_1(x):
    return x[:, 3:, :, :]

def add_text_to_image(image_tensor, text, font_size=15, font_color=(255, 0, 0)):
    if image_tensor.shape[1] == 1:
        image_tensor = image_tensor.repeat(1, 3, 1, 1)
    # Convert tensor to PIL Image
    image_pil = TF.to_pil_image(image_tensor.cpu()[0])
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default(font_size)
    draw.rectangle((3, 1, 40, 23), fill=(0, 0, 0))
    draw.text((5, 3), text, font_color, font=font)

    return TF.to_tensor(image_pil).unsqueeze(0)

def PairRandomCrop(img, crop_size):
    _, _, H, W = img.shape
    top = torch.randint(0, H - crop_size, (1,)).item()
    left = torch.randint(0, W - crop_size, (1,)).item()

    return TF.crop(img, top, left, crop_size, crop_size)

def expand_channel(x):
    return torch.cat((x, x, x), dim=1)
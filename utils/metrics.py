import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import Inception3
from torchvision import transforms
import lpips
 

transform_1 = transforms.Compose([
    transforms.Resize((256, 256))
])

transform_2 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


class PSNR():
    # Peak Signal-to-Noise Ratio
    def __init__(self):
        super(PSNR, self).__init__()

    def psnr(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

    def psnr_metric(self, img1, img2):
        img1 = transform_1(img1)
        img2 = transform_2(img2)
        img1 = img1.cuda()
        img2 = img2.cuda()
        return self.psnr(img1, img2)


class SSIM():
    # Structural Similarity Index
    def __init__(self):
        super(SSIM, self).__init__()
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def ssim(self, img1, img2):
        # Values range from 0 to 1
        img1 = transform_1(img1)
        img2 = transform_2(img2)
        img1 = img1.cuda()
        img2 = img2.cuda()
        mu1 = F.avg_pool2d(img1, 3, 1)
        mu2 = F.avg_pool2d(img2, 3, 1)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.avg_pool2d(img1 ** 2, 3, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 ** 2, 3, 1) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, 3, 1) - mu1_mu2

        # Values range from 0 to 1
        ssim_map = ((2 * mu1_mu2 + self.c1) * (2 * sigma12 + self.c2)) / ((mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2))
        return ssim_map

    def ssim_metric(self, img1, img2):
        ssim_map = self.ssim(img1, img2)
        ssim = torch.clamp((ssim_map), 0, 1)
        return torch.mean(ssim)

    def ssim_metric(self, img1, img2):
        ssim_map = self.ssim(img1, img2)
        ssim = torch.clamp((ssim_map), 0, 1)
        return torch.mean(ssim)


class PI():
    # Perceptual Index
    def __init__(self):
        super(PI, self).__init__()

    def pi(self, img1, img2):
        return torch.sum(img1 * img2) / torch.sqrt(torch.sum(img1 ** 2) * torch.sum(img2 ** 2))

    def pi_metric(self, img1, img2):
        return 1 - self.pi(img1, img2)


class LPIPS():
    # Learned Perceptual Image Patch Similarity
    def __init__(self):
        super(LPIPS, self).__init__()
        self.lpips = lpips.LPIPS(net='alex').cuda()

    def lpips_metric(self, img1, img2):
        img1 = transform_1(img1)
        img2 = transform_2(img2)
        img1 = img1.cuda()
        img2 = img2.cuda()
        return self.lpips(img1, img2).item()

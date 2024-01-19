import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import Inception3


class PSNR():
    # Peak Signal-to-Noise Ratio
    def __init__(self):
        super(PSNR, self).__init__()

    def psnr(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

    def psnr_loss(self, img1, img2):
        return -self.psnr(img1, img2)


class SSIM():
    # Structural Similarity Index
    def __init__(self):
        super(SSIM, self).__init__()
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def ssim(self, img1, img2):
        # Values range from 0 to 1
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

    def ssim_loss(self, img1, img2):
        ssim_map = self.ssim(img1, img2)
        return torch.clamp((1 - ssim_map) / 2, 0, 1)
    

class PI():
    # Perceptual Index
    def __init__(self):
        super(PI, self).__init__()

    def pi(self, img1, img2):
        return torch.sum(img1 * img2) / torch.sqrt(torch.sum(img1 ** 2) * torch.sum(img2 ** 2))

    def pi_loss(self, img1, img2):
        return -self.pi(img1, img2)


class FID():
    # Frechet Inception Distance
    def __init__(self):
        super(FID, self).__init__()
        self.inception = Inception3()
        self.inception.eval()
        self.inception.fc = nn.Sequential()
        self.inception.aux_logits = False
        self.inception.cuda()
    
    def torch_cov(self, m, rowvar=False):
        # Covariance matrix
        if m.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if m.dim() < 2:
            m = m.view(1, -1)
        if not rowvar and m.size(0) != 1:
            m = m.t()
        # m = m.type(torch.double)  # uncomment this line if desired
        fact = 1.0 / (m.size(1) - 1)
        m -= torch.mean(m, dim=1, keepdim=True)
        mt = m.t()

    def fid(self, img1, img2):
        # Values range from 0 to 1
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2
        with torch.no_grad():
            feat1 = self.inception(img1)[0].view(img1.size(0), -1)
            feat2 = self.inception(img2)[0].view(img2.size(0), -1)
            mean1, cov1 = torch.mean(feat1, dim=0), self.torch_cov(feat1, rowvar=False)
            mean2, cov2 = torch.mean(feat2, dim=0), self.torch_cov(feat2, rowvar=False)
            return torch.norm(mean1 - mean2) ** 2 + torch.trace(cov1 + cov2 - 2 * torch.sqrt(cov1 @ cov2))

    def fid_loss(self, img1, img2):
        return self.fid(img1, img2)
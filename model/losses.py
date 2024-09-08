import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class SmoothLossL():
    def __init__(self):
        super(SmoothLossL, self).__init__()

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)
    
    def smooth(self, input_I, input_R):
        input_R = 0.299*input_R[:, 0, :, :] + 0.587*input_R[:, 1, :, :] + 0.114*input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) + self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))
    

class SmoothLossR():
    def __init__(self):
        super(SmoothLossR, self).__init__()
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view(1, 1, 2, 2).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

    def gradient(self, input_tensor, direction):
        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y

        # Adjusting the kernel for 3-channel input
        channel = input_tensor.size(1)
        kernel = kernel.repeat(channel, 1, 1, 1)

        grad_out = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1, groups=channel))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction), kernel_size=3, stride=1, padding=1)
    
    def smooth(self, input_I, input_R):
        # Convert input_R to grayscale
        input_R = 0.299*input_R[:, 0, :, :] + 0.587*input_R[:, 1, :, :] + 0.114*input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)

        # Compute gradients and average gradients for both input_I and input_R
        grad_I_x = self.gradient(input_I, "x")
        grad_I_y = self.gradient(input_I, "y")
        ave_grad_R_x = self.ave_gradient(input_R, "x")
        ave_grad_R_y = self.ave_gradient(input_R, "y")

        # Compute the smoothness loss
        return torch.mean(grad_I_x * torch.exp(-10 * ave_grad_R_x) + grad_I_y * torch.exp(-10 * ave_grad_R_y))


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
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

    def forward(self, img1, img2):
        ssim_map = self.ssim(img1, img2)
        ssim = torch.clamp((1 - ssim_map) / 2, 0, 1)
        return torch.mean(ssim)



def create_gaussian_window(window_size, sigma):
    gauss = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    gauss = torch.exp(-(gauss ** 2) / (2 * sigma ** 2))
    gauss /= gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2D_window

class SSIMLoss_(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super(SSIMLoss_, self).__init__()
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2
        self.window_size = window_size
        self.sigma = sigma
        self.window = create_gaussian_window(window_size, sigma)
        self.first = True

    def ssim(self, img1, img2):
        _, channel, _, _ = img1.size()

        if self.first:
            self.window = self.window.to(img1.device).expand(channel, 1, self.window_size, self.window_size)
            self.first = False

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=channel)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.c1) * (2 * sigma12 + self.c2)) / ((mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2))
        return ssim_map

    def forward(self, img1, img2):
        ssim_map = self.ssim(img1, img2)
        ssim = torch.clamp((1 - ssim_map) / 2, 0, 1)
        out = torch.mean(ssim)
        return out
    

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        out = self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1 + 1e-8).mean()
        return out
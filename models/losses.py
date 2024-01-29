import torch
import torch.nn.functional as F


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


class SSIM():
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
        ssim = torch.clamp((1 - ssim_map) / 2, 0, 1)
        return torch.mean(ssim)
    

class PSNR():
    def __init__(self):
        super(PSNR, self).__init__()

    def psnr(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 10 * torch.log10(1 / mse)

    def psnr_loss(self, img1, img2):
        return -self.psnr(img1, img2)
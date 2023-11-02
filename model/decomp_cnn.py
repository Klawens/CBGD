import torch
import torch.nn as nn


class DecomNet(nn.Module):
    # URetinexNet
    def __init__(self):
        super().__init__()
        self.decom = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )

    def forward(self, x):
        output = self.decom(x)
        R = output[:, 0:3, :, :]    # Reflectance
        L = output[:, 3:4, :, :]    # Illumination
        R = torch.sigmoid(R)
        L = torch.sigmoid(L)
        
        return R, L

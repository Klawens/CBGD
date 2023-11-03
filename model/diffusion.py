import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

import cv2
import time

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


# Execute at each training step
class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T  # T diffusion steps
        # Generate a linear schedule 'betas' of length T from beta_1=1e-4 to beta_T=0.02
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas 
        alphas_bar = torch.cumprod(alphas, dim=0)   # cumprod: cumulative product

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # q(x_t | x_{t-1}) = N(x_t; sqrt(alpha_t) * x_{t-1}, (1 - alpha_t) * I)
        # x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * eps_t
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, idx, high):
        """
        Algorithm 1.
        """
        # x_0.shape = [b, c, h, w]
        # t = torch.randint(0, self.num_timesteps, (b,), device=device)
        # in which self.T is num_timesteps, b (batch) is x_0.shape[0], 0 is defalut low
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)   # randn_like generate normal distribution noise
        if high:
            noise = -torch.abs(noise)   # adding black noise, worked
        bg = torch.ones_like(noise)
        cv2.imwrite("./middle/noise/noise.jpg", (-torch.abs(noise)[0] + bg[0]).permute(1, 2, 0).cpu().numpy() * 255)

        # x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * eps_t
        # noise: eps_t ~ N(0, I)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        unet_out = self.model(x_t, t)   # predict noise
        loss = F.mse_loss(unet_out, noise, reduction='none')
        return loss, x_t, unet_out


# Execute once at evaluation
class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        # or replaced with
        # var = self.betas
        var = extract(var, t, x_t.shape)

        compile_eps = torch.compile(self.model)

        eps = compile_eps(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        x_t = x_T
        compile_p_mean_variance = torch.compile(self.p_mean_variance)
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= compile_p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
                # noise = torch.abs(noise)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            # assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)


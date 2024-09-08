import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

import numpy as np
from tqdm import tqdm


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    # print(out.shape, v.shape, t.shape)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        # self.I_loss = smoothloss()

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)  # (b, 4, h, w)
        h_1, h_2 = self.model(x_t, t)
        h = torch.cat((h_1, h_2), dim=1)
        loss = F.mse_loss(h, noise, reduction='none')

        return loss, x_t


class GaussianDiffusionSampler(nn.Module):
    '''
    DDPM Sampler.
    '''
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
        var = extract(var, t, x_t.shape)

        eps_1, eps_2 = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=torch.cat((eps_1, eps_2), dim=1))

        return xt_prev_mean, var

    def forward(self, x_T):
        x_t = x_T
        for time_step in reversed(range(self.T)):
            # print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)


class DDIMSampler(nn.Module):
    def __init__(self, model, beta, T, steps=100, method="linear", eta=0.0,
                only_return_x_0=True, interval=1):
        super().__init__()
        self.model = model
        self.T = T
        self.steps = steps
        self.method = method
        self.eta = eta
        self.only_return_x_0 = only_return_x_0
        self.interval = interval

        # generate T steps of beta
        beta_t = torch.linspace(beta[0], beta[1], T, dtype=torch.float32)
        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - beta_t
        self.register_buffer("alpha_t_bar", torch.cumprod(alpha_t, dim=0))

    def sample_one_step(self, x_t, time_step, prev_time_step, eta):
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        # get current and previous alpha_cumprod
        alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape)

        # predict noise using model
        epsilon_theta_t_ = self.model(x_t, t)
        epsilon_theta_t = torch.cat(epsilon_theta_t_, dim=1)

        # calculate x_{t-1}
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        epsilon_t = torch.randn_like(x_t)

        x_t_minus_one = (
            torch.sqrt(alpha_t_prev / alpha_t) * x_t + (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt((alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t + sigma_t * epsilon_t)

        return x_t_minus_one

    def forward(self, x_t):
        """
        Parameters:
            x_t: Standard Gaussian noise. A tensor with shape (batch_size, channels, height, width).
            steps: Sampling steps, actuall steps.
            method: Sampling method, can be "linear" or "quadratic".
            eta: Coefficients of sigma parameters in the paper. The value 0 indicates DDIM, 1 indicates DDPM.
            only_return_x_0: Determines whether the image is saved during the sampling process. if True,
                intermediate pictures are not saved, and only return the final result $x_0$.
            interval: This parameter is valid only when `only_return_x_0 = False`. Decide the interval at which
                to save the intermediate process pictures, according to `step`.
                $x_t$ and $x_0$ will be included, no matter what the value of `interval` is.

        Returns:
            if `only_return_x_0 = True`, will return a tensor with shape (batch_size, channels, height, width),
            otherwise, return a tensor with shape (batch_size, sample, channels, height, width),
            include intermediate pictures.
        """
        if self.method == "linear":
            a = self.T // self.steps
            time_steps = np.asarray(list(range(0, self.T, a)))
        elif self.method == "quadratic":
            time_steps = (np.linspace(0, np.sqrt(self.T * 0.8), self.steps) ** 2).astype(np.int)
        else:
            raise NotImplementedError(f"sampling method {self.method} is not implemented!")

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        time_steps = time_steps + 1
        # previous sequence
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])

        x = [x_t]

        for i in reversed(range(0, self.steps)):
            x_t = self.sample_one_step(x_t, time_steps[i], time_steps_prev[i], self.eta)
            
            # save_image(x_t[:, :3, :, :] * x_t[:, 3:, :, :], f"./visualization/temp_{i}.jpg")
            # save_image(x_t[:, :3, :, :], f"./visualization/temp_{i}_r.jpg")
            # save_image(x_t[:, 3:, :, :], f"./visualization/temp_{i}_i.jpg")

            if not self.only_return_x_0 and ((self.steps - i) % self.interval == 0 or i == 0):
                x.append(torch.clip(x_t, -1.0, 1.0))

        if self.only_return_x_0:
            return x_t  # [batch_size, channels, height, width]
            # return torch.clip(x_t, -1.0, 1.0)
        return torch.stack(x, dim=1)  # [batch_size, sample, channels, height, width]
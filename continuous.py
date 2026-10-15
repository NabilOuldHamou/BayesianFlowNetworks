import numpy as np
import torch
import torch.nn as nn


def bcast(t, x):
    return t.view(-1, *[1] * (x.dim() - 1))

class ContinuousBFN(nn.Module):
    def __init__(self, net, sigma1=1e-3, t_min=1e-6):
        super().__init__()
        self.net = net
        self.sigma1 = sigma1
        self.t_min = t_min

    def gamma(self, t):
        return 1 - self.sigma1 ** (2 * t)

    def x_hat(self, mu, t):
        tb = bcast(t, mu)
        g = self.gamma(tb).clamp_min(1e-12)
        eps = self.net(mu, t)
        x = (mu / g - torch.sqrt((1 - g) / g) * eps).clamp(-1, 1)
        return torch.where(tb < self.t_min, torch.zeros_like(x), x)

    def loss(self, x):
        t = torch.rand(x.shape[0], device=x.device)
        tb = bcast(t, x); g = self.gamma(tb)
        mu = g * x + torch.sqrt(g * (1 - g)) * torch.randn_like(x)
        err = (x - self.x_hat(mu, t)).pow(2).flatten(1).sum(1)
        return (-np.log(self.sigma1) * self.sigma1 ** (-2 * t) * err).mean()

    @torch.no_grad()
    def sample(self, shape, n=100, device="cpu"):
        mu, rho = torch.zeros(shape, device=device), 1.0
        for i in range(1, n + 1):
            t = torch.full((shape[0],), (i - 1) / n, device=device)
            xh = self.x_hat(mu, t)
            alpha = self.sigma1 ** (-2 * i / n) * (1 - self.sigma1 ** (2 / n))
            y = xh + torch.randn_like(xh) / np.sqrt(alpha)
            mu = (rho * mu + alpha * y) / (rho + alpha); rho += alpha
        return self.x_hat(mu, torch.ones(shape[0], device=device))
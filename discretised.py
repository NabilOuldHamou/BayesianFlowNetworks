import numpy as np
import torch
from continuous import ContinuousBFN, bcast

class DiscretisedBFN(ContinuousBFN):
    def __init__(self, net, K=256, sigma1=1e-3, t_min=1e-6):
        super().__init__(net, sigma1, t_min)
        self.K = K
        self.register_buffer("edges", torch.linspace(-1, 1, K + 1))
        self.register_buffer("centres", (self.edges[:-1] + self.edges[1:]) / 2)

    def probs(self, mu, t):
        tb = bcast(t, mu)
        g = self.gamma(tb).clamp_min(1e-12)
        out = self.net(mu, t); mu_eps, ls_eps = out[..., 0], out[..., 1]
        r = torch.sqrt((1 - g) / g)
        mu_x = torch.where(tb < self.t_min, torch.zeros_like(mu), mu / g - r * mu_eps)
        sd_x = torch.where(tb < self.t_min, torch.ones_like(mu), r * ls_eps.exp())
        z = (self.edges - mu_x.unsqueeze(-1)) / sd_x.unsqueeze(-1)
        cdf = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf[..., 1:-1], torch.ones_like(cdf[..., :1])], -1)
        return (cdf[..., 1:] - cdf[..., :-1]).clamp_min(0)

    def loss(self, x):
        t = torch.rand(x.shape[0], device=x.device)
        tb = bcast(t, x); g = self.gamma(tb)
        mu = g * x + torch.sqrt(g * (1 - g)) * torch.randn_like(x)
        k_hat = (self.probs(mu, t) * self.centres).sum(-1)
        err = (x - k_hat).pow(2).flatten(1).sum(1)
        return (-np.log(self.s1) * self.s1 ** (-2 * t) * err).mean()

    @torch.no_grad()
    def sample(self, shape, n=100, device="cpu"):
        mu, rho = torch.zeros(shape, device=device), 1.0
        for i in range(1, n + 1):
            t = torch.full((shape[0],), (i - 1) / n, device=device)
            k = torch.distributions.Categorical(probs=self.probs(mu, t)).sample()
            alpha = self.s1 ** (-2 * i / n) * (1 - self.s1 ** (2 / n))
            y = self.centres[k] + torch.randn(shape, device=device) / np.sqrt(alpha)
            mu = (rho * mu + alpha * y) / (rho + alpha); rho += alpha
        k = torch.distributions.Categorical(probs=self.probs(mu, torch.ones(shape[0], device=device))).sample()
        return self.centres[k]
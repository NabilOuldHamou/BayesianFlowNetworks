import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from continuous import bcast

class DiscreteBFN(nn.Module):
    def __init__(self, net, K, beta1):
        super().__init__()
        self.net = net
        self.K = K
        self.beta1 = beta1

    def p_out(self, theta, t):
        return F.softmax(self.net(2 * theta - 1, t), -1)

    def flow(self, x, t):
        e_x = F.one_hot(x, self.K).float()
        beta = self.beta1 * bcast(t, e_x) ** 2
        y = beta * (self.K * e_x - 1) + (beta * self.K).sqrt() * torch.randn_like(e_x)
        return F.softmax(y, -1), e_x

    def loss(self, x):
        t = torch.rand(x.shape[0], device=x.device)
        theta, e_x = self.flow(x, t)
        e_hat = self.p_out(theta, t)
        err = (e_x - e_hat).pow(2).flatten(1).sum(1)
        return (self.K * self.beta1 * t * err).mean()

    @torch.no_grad()
    def sample(self, B, D, n=100, device="cpu"):
        theta = torch.full((B, D, self.K), 1 / self.K, device=device)
        for i in range(1, n + 1):
            t = torch.full((B,), (i - 1) / n, device=device)
            k = torch.distributions.Categorical(probs=self.p_out(theta, t)).sample()
            alpha = self.beta1 * (2 * i - 1) / n ** 2
            e_k = F.one_hot(k, self.K).float()
            y = alpha * (self.K * e_k - 1) + np.sqrt(alpha * self.K) * torch.randn_like(e_k)
            theta = F.softmax(y + theta.clamp_min(1e-30).log(), -1)
        return self.p_out(theta, torch.ones(B, device=device)).argmax(-1)

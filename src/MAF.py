import math

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


class MADE_Layer(nn.Linear):
    def __init__(self, in_features, out_features, conditional_dim=None) -> None:
        super().__init__(
            in_features,
            out_features,
        )
        self.register_buffer("mask", None)

        if conditional_dim is not None:
            self.conditional_linear = nn.Linear(
                conditional_dim, out_features, bias=False
            )
        nn.init.zeros_(self.bias)

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, X, cond=None):
        val = F.linear(X, self.mask * self.weight, self.bias)
        if cond is None:
            return val
        else:
            return val + self.conditional_linear(cond)


class MADE(nn.Module):
    def __init__(self, n_in, n_out, sizes, cond_linear, mixture=None) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = 2 * n_out if mixture is None else 3 * n_out * mixture
        self.hidden_dims = sizes
        self.masks = {}
        self.layers = []
        dim_list = [n_in, *sizes, self.n_out]
        self.pre = MADE_Layer(dim_list[0], dim_list[1], cond_linear)
        for i in range(1, len(dim_list) - 2):
            self.layers.append(MADE_Layer(dim_list[i], dim_list[i + 1]))
            self.layers.append(nn.GELU())
        self.layers.append(MADE_Layer(dim_list[-2], dim_list[-1]))
        self.model = nn.Sequential(*self.layers)
        self.dim_list = dim_list
        self.mixture = 0 if mixture is None else mixture
        self.get_masks()

    def get_masks(self) -> None:
        L = len(self.hidden_dims)

        for i in range(L + 1):
            if i == 0:
                in_degrees = torch.arange(self.n_in) % self.n_in
            else:
                in_degrees = torch.arange(self.dim_list[i]) % (self.n_in - 1)

            if i == L:
                out_degrees = torch.arange(self.dim_list[i + 1]) % self.n_in - 1

            else:
                out_degrees = torch.arange(self.dim_list[i + 1]) % (self.n_in - 1)

            self.masks[i] = (
                out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)
            ).float()

        for i in range(L + 1):
            if i == 0:
                self.pre.set_mask(self.masks[i])
            else:
                self.model[2 * (i - 1)].set_mask(self.masks[i])

    def forward(self, X, X_cond):
        X = F.gelu(self.pre(X, X_cond))
        return self.model(X)


class MAFBlock(nn.Module):
    def __init__(self, n_in, n_out, hidden_dims, cond_dim, min=1e-3, max=3.0) -> None:
        super().__init__()
        self.made = MADE(n_in, n_out, hidden_dims, cond_dim)
        self.n = len(hidden_dims) + 2
        self.ratio = 0.5
        self.max = max
        self.min = min
        self.beta = np.log(2)

    def forward(self, X, cond):
        out = self.made(X, cond)
        mu, alpha = torch.chunk(out, 2, dim=-1)
        scale = torch.exp(-self.ratio * alpha)
        # scale = F.softplus(self.ratio * alpha,beta = self.beta)
        # scale = -F.logsigmoid(alpha)
        # scale = torch.abs(alpha) + self.min
        # scale = torch.clamp(scale,self.min,self.max)
        scale = torch.clamp(scale, self.min, self.max)
        u = (X - mu) * scale
        return u, torch.log(scale)

    def sample(self, u, cond):
        X = torch.zeros_like(u)
        for i in range(X.shape[-1]):
            out = self.made(X, cond)
            mu, alpha = torch.chunk(out, 2, dim=-1)
            scale = torch.exp(-self.ratio * alpha[:, :, i])
            # scale = F.softplus(self.ratio * alpha[:,:,i],beta = self.beta)
            scale = torch.clamp(scale, self.min, self.max)
            X[:, :, i] = mu[:, :, i] + u[:, :, i] / scale
        return X, -self.ratio * alpha


class MAFBlock_Mix(nn.Module):
    def __init__(
        self, n_in, n_out, hidden_dims, cond_dim, n_comp, min=1e-4, max=4.0
    ) -> None:
        super().__init__()
        self.made = MADE(n_in, n_out, hidden_dims, cond_dim, mixture=n_comp)
        self.n = len(hidden_dims) + 2
        self.ratio = 0.5
        self.n_comp = n_comp
        self.max = max
        self.min = min
        self.beta = np.log(2)

    def forward(self, X, cond):
        B, L = X.shape
        out = self.made(X, cond).reshape(B, self.n_comp, 3 * L)
        mu, alpha, mix_w = torch.chunk(out, 3, dim=-1)
        X = X.reshape(B, 1, L).repeat_interleave(self.n_comp, 1)
        scale = torch.exp(-self.ratio * alpha)
        # scale = F.softplus(self.ratio * alpha,beta = self.beta)
        # scale = torch.abs(alpha) + self.min
        # scale = torch.clamp(scale,self.min,self.max)
        scale = torch.clamp(scale, self.min, self.max)
        u = (X - mu) * scale
        log_det = mix_w - torch.logsumexp(mix_w, dim=1, keepdim=True) + torch.log(scale)
        return u, log_det

    def sample(self, u, cond):
        B, N, C, D = u.shape
        X = torch.zeros([B, N, D], device=u.device)
        for i in range(X.shape[-1]):
            out = self.made(X, cond).reshape(B, N, C, 3 * D)
            mu, alpha, mix_w = torch.chunk(out, 3, dim=-1)
            mix_w = mix_w - torch.logsumexp(mix_w, dim=2, keepdim=True)
            z = (
                torch.distributions.Categorical(logits=mix_w[:, :, :, i])
                .sample()
                .unsqueeze(-1)
            )
            u_selected = torch.gather(u[:, :, :, i], 2, z)
            mu_selected = torch.gather(mu[:, :, :, i], 2, z)
            alpha_selected = torch.gather(alpha[:, :, :, i], 2, z)
            scale = torch.exp(-self.ratio * alpha_selected).squeeze(-1)
            # scale = F.softplus(self.ratio * alpha[:,:,i],beta = self.beta)
            scale = torch.clamp(scale, self.min, self.max)
            X[:, :, i] = mu_selected.squeeze(-1) + u_selected.squeeze(-1) / scale
        log_det = mix_w - self.ratio * alpha
        return X, log_det


class Shuffle(nn.Module):
    def __init__(self, X_dim) -> None:
        super().__init__()
        perm = np.random.permutation(X_dim)
        inv_perm = np.argsort(perm)
        self.register_buffer("perm", torch.from_numpy(perm))
        self.register_buffer("inv_perm", torch.from_numpy(inv_perm))

    def forward(self, X, cond):
        return X[:, self.perm], torch.zeros_like(X)

    def sample(self, X, cond):
        return X[:, :, self.inv_perm], torch.zeros_like(X)


class Rotate(nn.Module):
    def __init__(self, X_dim) -> None:
        super().__init__()
        A = np.random.randn(X_dim, X_dim)
        M = (A + A.T) / np.sqrt(2)
        eig, U = np.linalg.eig(M)
        self.register_buffer("rot", torch.from_numpy(U).float())
        self.register_buffer("inv_rot", torch.from_numpy(np.linalg.inv(U)).float())

    def forward(self, X, cond):
        return F.linear(X, self.rot), torch.zeros_like(X)

    def sample(self, X, cond):
        return F.linear(X, self.inv_rot), torch.zeros_like(X)


class Normalize(nn.Module):
    def __init__(self, X_dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros([X_dim]))
        self.register_buffer("std", torch.ones([X_dim]))

    def set_means(self, mean, std):
        self.mean = mean
        self.std = std

    def forward(self, X, cond):
        X = (X - self.mean) / self.std
        return X, torch.zeros_like(X) - torch.log(self.std)

    def sample(self, u, cond):
        X = u * self.std + self.mean
        return X, torch.zeros_like(u) - torch.log(self.std)


class ConvertAV(nn.Module):
    def __init__(self, dim=3, a=0.5, b=4.0):
        super().__init__()
        self.dim = dim
        self.register_buffer("a", torch.tensor(a))
        self.register_buffer("b", torch.tensor(b))

    def forward(self, X, cond):
        X[:, self.dim] = (
            1 / self.a * torch.log(torch.exp(X[:, self.dim] * self.a) - 1) * 1 / self.b
        )
        det_log = torch.zeros_like(X)
        det_log[:, self.dim] = (
            0  # self.a * (F.softplus(X[:,self.dim],beta = self.a) - X[:,self.dim])
        )
        return X, det_log

    def sample(self, u, cond):
        u[:, :, self.dim] = F.softplus(u[:, :, self.dim] * self.b, beta=self.a)
        det_log = torch.zeros_like(u)
        det_log[:, :, self.dim] = (
            0  # self.a * (F.softplus(X[:,self.dim] - X[:,self.dim]))
        )
        return u, det_log


class MAF(nn.Module):
    def __init__(
        self,
        N,
        hidden_dims,
        X_dim,
        cond_dim,
        n_comp,
        preprocess=None,
        type_shuff="r",
        **kwargs,
    ) -> None:
        super().__init__()
        min = 0.2
        max = 10.0
        self.net = [Normalize(X_dim), ConvertAV()]
        shuffle = Shuffle if type_shuff == "s" else Rotate
        for _ in range(N - 1):
            self.net.append(shuffle(X_dim))
            self.net.append(
                MAFBlock(X_dim, X_dim, hidden_dims, cond_dim, min=min, max=max)
            )
        self.net.append(shuffle(X_dim))
        self.net.append(
            MAFBlock_Mix(
                X_dim, X_dim, hidden_dims, cond_dim, n_comp=n_comp, min=min, max=max
            )
        )
        self.net = nn.ModuleList(self.net)
        self.N = N
        self.X_dim = X_dim
        self.preprocess = preprocess
        self.n_comp = n_comp

    def forward(self, X, cond):
        if self.preprocess is not None:
            cond = self.preprocess(cond, Y)
        det_prob = torch.zeros([X.shape[0], self.n_comp, self.X_dim]).to(X.device)
        for layer in self.net:
            X, prob = layer(X, cond)
            if len(prob.shape) == 2:
                prob = prob.unsqueeze(1)
            det_prob += prob
        log_probs = -0.5 * X.pow(2) - 0.5 * math.log(2 * math.pi)
        probs = torch.logsumexp(log_probs + det_prob, dim=1)
        return torch.mean(probs, axis=1)

    @torch.no_grad()
    def sample(self, cond, how_many, Y=None, return_log_prob=False):
        if self.preprocess is not None:
            if Y is None:
                cond = self.preprocess(cond)
            else:
                cond = self.preprocess(cond, Y)
        u = torch.randn([cond.shape[0], how_many, self.n_comp, self.X_dim]).to(
            cond.device
        )
        log_probs = -0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)
        cond = cond.unsqueeze(1).repeat_interleave(how_many, 1)
        det_prob = torch.zeros([cond.shape[0], how_many, self.n_comp, self.X_dim]).to(
            cond.device
        )
        for layer in reversed(self.net):
            u, prob = layer.sample(u, cond)
            if len(prob.shape) == 3:
                prob = prob.unsqueeze(2)
            det_prob += prob
        if return_log_prob:
            return u, torch.mean(torch.logsumexp(log_probs + det_prob, dim=2), dim=-1)
        return u

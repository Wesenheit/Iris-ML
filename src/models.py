import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import distributions as D
import numpy as np
from typing import Optional, List
from MAF import MAF


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            num_heads=num_heads, embed_dim=d_model, dropout=dropout
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.GELU(),
        )

    def forward(self, x, Q=None):
        res_stream = self.norm1(x)
        if Q is None:
            Q = x
        res_stream = self.mha(Q, res_stream, res_stream, need_weights=False)
        x = x + res_stream[0]
        res_stream = self.norm2(x)
        x = x + self.ff(res_stream)
        return x


class SEDTransformer(nn.Module):
    def __init__(
        self,
        bands: int,
        d_model: int,
        heads: int,
        hidden: int,
        dropout: float,
        layers: int,
        **kwargs,
    ) -> None:
        """
        d_model - dimensionality of model
        num_heads - number of heads
        hidden - number of neurons in feedforward network
        droput - self-explenatory
        N_layers - numbers of layers in the transformer
        """
        super().__init__()
        self.d_model = d_model
        self.network = [
            TransformerBlock(
                d_model=d_model, num_heads=heads, hidden=hidden, dropout=dropout
            )
            for _ in range(layers)
        ]
        self.network = nn.Sequential(*self.network)
        self.embedding = nn.Embedding(bands, d_model)
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.mlp_head = nn.Linear(d_model, 5)
        self.pre = nn.Linear(2, d_model, bias=False)
        self.register_buffer("means", torch.zeros([bands]))
        self.register_buffer("stds", torch.ones([bands]))

    def forward(self, bands: torch.tensor, mags: torch.tensor, errors=None):
        X = self.pass_token(bands, mags, errors)
        return torch.exp(self.mlp_head(X))

    def pass_token(self, bands: torch.tensor, mags: torch.tensor, errors=None):
        emb_tokens = self.embedding(bands)
        S, B, _ = emb_tokens.shape
        means = self.means[bands]
        stds = self.stds[bands]
        mags_n = mags - torch.mean(mags, axis=0, keepdim=True)
        mags_n = (mags_n - means) / stds
        if errors is None:
            errors_n = torch.zeros_like(mags)
        else:
            errors_n = errors.clone() * 100
            # errors_n = errors_n/torch.mean(errors_n,dim = 0,keepdim = True)
            # errors_n = torch.zeros_like(mags)

        mags_n = mags_n.reshape(S, B, -1)
        errors_n = errors_n.reshape(S, B, -1)
        out = torch.cat([mags_n, errors_n], dim=-1)
        out = self.pre(out)
        emb_tokens = emb_tokens + out
        cls_token = self.class_token.repeat(1, B, 1)
        X = torch.cat([cls_token, emb_tokens], dim=0)
        X = self.network(X)
        return X[0]

    def jackknife(self, bands: torch.tensor, mags: torch.tensor):
        eye = torch.eye(len(bands), device=bands.device).bool()
        list_of_bands = []
        list_of_mags = []
        for i in range(len(bands)):
            idx = torch.logical_not(eye[:, i])
            list_of_bands.append(bands[idx])
            list_of_mags.append(mags[idx])
        new_mags = torch.stack(list_of_mags).T
        new_bands = torch.stack(list_of_bands).T
        return self(new_bands, new_mags)


class SED_NDE(nn.Module):
    def __init__(self, MAF_dict, Transformer_dict):
        super().__init__()
        self.MAF = MAF(**MAF_dict)
        self.Transformer = SEDTransformer(**Transformer_dict)
        self.preprocess = nn.Sequential(
            nn.Linear(Transformer_dict["d_model"], MAF_dict["cond_dim"]), nn.GELU()
        )
        self.embed_err = nn.Linear(1, Transformer_dict["d_model"], bias=False)

        self.loglike = lambda X: torch.where(
            X[:, :, -1] < 0,
            torch.zeros(X.shape[:-1]).to(X.device),
            torch.ones(X.shape[:-1]).to(X.device),
        )

    def forward(
        self,
        target: torch.tensor,
        bands: torch.tensor,
        mags: torch.tensor,
        errors,
        dev="cuda",
    ):
        X_out = self.Transformer.pass_token(bands, mags, errors)
        X_out = self.preprocess(X_out)
        log_prob = self.MAF(target, X_out)
        return log_prob

    @torch.no_grad()
    def sample(
        self,
        how_many: int,
        bands: torch.tensor,
        mags: torch.tensor,
        errors=None,
        dev="cuda",
        **kwargs,
    ):
        X_out = self.Transformer.pass_token(bands, mags, errors)
        X_out = self.preprocess(X_out)
        samples = self.MAF.sample(X_out, how_many, **kwargs)
        return samples

    def get_weights(self, samples):
        raise NotImplementedError

    @torch.no_grad()
    def sample_SIR(
        self, how_many: int, bands: torch.tensor, mags: torch.tensor, errors=None
    ):
        raise NotImplementedError

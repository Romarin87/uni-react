import math
from typing import Optional

import torch


class RBFEmb(torch.nn.Module):
    def __init__(self, num_rbf: int, soft_cutoff_upper: float):
        super().__init__()
        self.soft_cutoff_upper = soft_cutoff_upper
        self.soft_cutoff_lower = 0.0
        self.num_rbf = num_rbf

        means, betas = self._initial_params()
        self.register_buffer("means", means)
        self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.soft_cutoff_upper))
        end_value = torch.exp(torch.scalar_tensor(-self.soft_cutoff_lower))
        means = torch.linspace(start_value, end_value, self.num_rbf)
        betas = torch.tensor(
            [(2.0 / self.num_rbf * (end_value - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self) -> None:
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.unsqueeze(dim=-1)
        soft_cutoff = dist * torch.pi / self.soft_cutoff_upper
        soft_cutoff = 0.5 * (torch.cos(soft_cutoff) + 1.0)
        soft_cutoff[dist >= self.soft_cutoff_upper] = 0
        bias_sq = torch.square(torch.exp(-dist) - self.means)
        return soft_cutoff * torch.exp(-self.betas * bias_sq)


class NonLinear(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0, hidden: Optional[int] = None):
        super().__init__()
        if hidden is None:
            hidden = input_size
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DropPath(torch.nn.Module):
    def __init__(self, prob: float = 0.0):
        super().__init__()
        self.drop_prob = prob
        if not (0.0 <= prob <= 1.0):
            raise ValueError("dropout prob should be in [0, 1]")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class BiasedAttention(torch.nn.Module):
    def __init__(self, q_dim: int, k_dim: int, v_dim: int, head_dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        total_dim = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim

        self.drop_fn = torch.nn.Dropout(dropout)
        self.ln_q = torch.nn.Linear(q_dim, total_dim, bias=False)
        self.ln_k = torch.nn.Linear(k_dim, total_dim, bias=False)
        self.ln_v = torch.nn.Linear(v_dim, total_dim, bias=False)
        self.ln_o = torch.nn.Linear(total_dim, q_dim)
        self.lin_g = torch.nn.Linear(q_dim, total_dim)
        self.temperature = math.sqrt(head_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        bs, ql = q.shape[:2]
        kl = k.shape[1]
        vl = v.shape[1]
        if kl != vl:
            raise ValueError("key and values are not matched")

        qh = self.ln_q(q).reshape((bs, ql, self.heads, -1)) / self.temperature
        kh = self.ln_k(k).reshape((bs, kl, self.heads, -1))
        vh = self.ln_v(v).reshape((bs, vl, self.heads, -1))

        qk_dot = torch.einsum("bqhd,bkhd->bkqh", qh, kh)
        attn = qk_dot + attn_bias.transpose(1, 2)

        if key_padding_mask is not None:
            if key_padding_mask.shape != (bs, kl):
                raise ValueError(f"shape of key_padding_mask should be ({bs}, {kl})")
            attn[key_padding_mask] = float("-inf")

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                if attn_mask.shape != (ql, kl):
                    raise ValueError(f"shape of attn_mask should be ({ql}, {kl})")
                attn[:, attn_mask.transpose(0, 1)] = float("-inf")
            elif attn_mask.ndim == 3:
                if attn_mask.shape != (bs, ql, kl):
                    raise ValueError(f"shape of attn_mask should be ({bs}, {ql}, {kl})")
                attn[attn_mask.transpose(-1, -2)] = float("-inf")
            elif attn_mask.ndim == 4:
                if attn_mask.shape != (bs, ql, kl, self.heads):
                    raise ValueError(
                        f"shape of attn_mask should be ({bs}, {ql}, {kl}, {self.heads})"
                    )
                attn[attn_mask.transpose(1, 2)] = float("-inf")
            else:
                raise ValueError("available attn_mask dims are 2, 3 or 4")

        attn = self.drop_fn(torch.softmax(attn, dim=1))
        o = torch.einsum("bkqh,bkhd->bqhd", attn, vh).reshape((bs, ql, -1))
        o = torch.sigmoid(self.lin_g(q)) * o
        return self.ln_o(o), attn.transpose(1, 2)


class UniMolLayer(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        activation_dropout: float = 0.1,
        attn_dropout: float = 0.1,
        path_dropout: float = 0.1,
        ffn_dim: Optional[int] = None,
    ):
        super().__init__()
        self.attn = BiasedAttention(
            q_dim=dim,
            k_dim=dim,
            v_dim=dim,
            head_dim=dim // num_heads,
            heads=num_heads,
            dropout=attn_dropout,
        )
        self.dim = dim
        self.ffn_dim = (dim << 1) if ffn_dim is None else ffn_dim

        self.ffn = NonLinear(
            input_size=self.dim,
            output_size=self.dim,
            hidden=self.ffn_dim,
            dropout=activation_dropout,
        )
        self.drop_fn = DropPath(path_dropout)
        self.attn_ln = torch.nn.LayerNorm(dim)
        self.ffn_ln = torch.nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_bias: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        delta_x, edge_bias = self.attn(
            q=x,
            k=x,
            v=x,
            attn_bias=edge_bias,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
        )
        x = self.attn_ln(x + self.drop_fn(delta_x))
        x = self.ffn_ln(x + self.drop_fn(self.ffn(x)))
        return x, edge_bias


def safe_normalization(x: torch.Tensor, dim: int = -1, eps: float = 1e-16) -> torch.Tensor:
    x_norm2 = torch.sum(x * x, dim=dim, keepdim=True)
    return x / torch.sqrt(x_norm2 + eps)


# Backward-compatible typo alias – new code should use BiasedAttention.
BaisedAttention = BiasedAttention

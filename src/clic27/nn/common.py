"""Building blocks shared by DC-AE and the SANA DiT, ported from diffusers 0.40.0.

Attribute names match diffusers so its state dicts load with strict=True. Numerics (dtype casts,
eps, op order) follow the diffusers code; `experiments/parity/` checks outputs against diffusers.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """diffusers.models.normalization.RMSNorm (non-NPU path), normalising the last dim."""

    def __init__(self, dim: int, eps: float, elementwise_affine: bool = True, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None
        self.bias = nn.Parameter(torch.zeros(dim)) if elementwise_affine and bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight is None:
            return x.to(input_dtype)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


def channel_rms_norm(norm: RMSNorm, x: torch.Tensor) -> torch.Tensor:
    """Apply RMSNorm over the channel dim of an NCHW tensor."""
    return norm(x.movedim(1, -1)).movedim(-1, 1)


ACTIVATIONS = {"silu": nn.SiLU, "swish": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU, "mish": nn.Mish}


def get_activation(name: str) -> nn.Module:
    return ACTIVATIONS[name.lower()]()


class GLUMBConv(nn.Module):
    """diffusers.models.transformers.sana_transformer.GLUMBConv."""

    def __init__(self, in_channels: int, out_channels: int, expand_ratio: float = 4,
                 norm_type: str | None = None, residual_connection: bool = True):
        super().__init__()
        hidden = int(expand_ratio * in_channels)
        self.norm_type = norm_type
        self.residual_connection = residual_connection
        self.nonlinearity = nn.SiLU()
        self.conv_inverted = nn.Conv2d(in_channels, hidden * 2, 1, 1, 0)
        self.conv_depth = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2)
        self.conv_point = nn.Conv2d(hidden, out_channels, 1, 1, 0, bias=False)
        self.norm = RMSNorm(out_channels, eps=1e-5, elementwise_affine=True, bias=True) if norm_type == "rms_norm" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.nonlinearity(self.conv_inverted(x))
        x, gate = torch.chunk(self.conv_depth(x), 2, dim=1)
        x = self.conv_point(x * self.nonlinearity(gate))
        if self.norm_type == "rms_norm":
            x = channel_rms_norm(self.norm, x)
        return x + residual if self.residual_connection else x


def linear_attention(q: torch.Tensor, k_t: torch.Tensor, v: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    """ReLU linear attention as in SANA. q, v: (..., d, N); k_t: (..., N, d). Returns (..., d, N) in fp32."""
    v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1.0)  # extra row of ones -> normaliser
    out = torch.matmul(torch.matmul(v, k_t), q)
    out = out.to(torch.float32)
    return out[..., :-1, :] / (out[..., -1:, :] + eps)

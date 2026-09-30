"""Integer networks for entropy parameters (Ballé et al., ICLR 2019, simplified to power-of-two scales).

Anything that decides a CDF (hyper-synthesis h_s, context models) must give bit-identical outputs on
the encoder machine and on the CLIC server. A0 showed float32/bf16 differ between GPU and CPU (~80
indices per 8M), while "integers carried in float64" match everywhere: every product and partial sum
is an integer below 2**53, so the result does not depend on summation order or kernel choice.

Fixed-point convention: a tensor with `frac` fractional bits stores round-down(v * 2**frac) as integers.
- QConv2d weights: int8 with W fractional bits; bias: integer at (W + in_frac) fractional bits.
- Hidden activations (ReLU): out = clamp(floor(acc / 2**shift), 0, 2**a_bits - 1),
  shift = W + in_frac - out_frac (must be >= 0).
- Last layer (act=None) returns the raw accumulator with (W + in_frac) fractional bits.

Training uses the same quantisers with straight-through gradients (`forward`); decoding uses
`forward_int` on integer inputs. Run `forward_int` on CPU (default) to avoid any GPU kernel choice.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def ste(x_q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Value of x_q, gradient of x."""
    return x + (x_q - x).detach()


class QConv2d(nn.Conv2d):
    def __init__(self, cin, cout, k, stride=1, *, in_frac: int, out_frac: int | None = None, act: str | None = "relu",
                 w_frac: int = 7, w_max: int = 127, a_bits: int = 16, groups: int = 1):
        super().__init__(cin, cout, k, stride, padding=k // 2, groups=groups, bias=True)
        self.in_frac, self.w_frac, self.w_max, self.a_bits, self.act = in_frac, w_frac, w_max, a_bits, act
        self.out_frac = out_frac if act else w_frac + in_frac
        assert act in (None, "relu") and self.w_frac + self.in_frac >= self.out_frac

    @property
    def acc_frac(self) -> int:
        return self.w_frac + self.in_frac

    def _wq(self):
        return torch.clamp(torch.round(self.weight * 2 ** self.w_frac), -self.w_max, self.w_max)

    def _bq(self):
        return torch.round(self.bias * 2 ** self.acc_frac)

    def forward(self, x):  # float in/out, quantisers with STE
        w = ste(self._wq() / 2 ** self.w_frac, self.weight)
        b = ste(self._bq() / 2 ** self.acc_frac, self.bias)
        y = F.conv2d(x, w, b, self.stride, self.padding, 1, self.groups)
        if self.act is None:
            return y
        y = F.relu(y)
        q = torch.clamp(torch.floor(y * 2 ** self.out_frac), 0, 2 ** self.a_bits - 1) / 2 ** self.out_frac
        return ste(q, y)

    @torch.no_grad()
    def forward_int(self, x_int: torch.Tensor) -> torch.Tensor:  # float64 tensors holding integers
        w = self._wq().to(torch.float64)
        b = self._bq().to(torch.float64)
        acc = F.conv2d(x_int.to(torch.float64), w.to(x_int.device), b.to(x_int.device),
                       self.stride, self.padding, 1, self.groups)
        if self.act is None:
            return acc
        shift = self.acc_frac - self.out_frac
        return torch.clamp(torch.floor(F.relu(acc) / 2 ** shift), 0, 2 ** self.a_bits - 1)


class Up2(nn.Module):
    """Nearest-neighbour x2 upsampling (exact on integers)."""

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="nearest")

    forward_int = forward


class IntSequential(nn.Sequential):
    def forward_int(self, x_int: torch.Tensor) -> torch.Tensor:
        for m in self:
            x_int = m.forward_int(x_int)
        return x_int

    @property
    def out_frac(self) -> int:
        return [m for m in self if isinstance(m, QConv2d)][-1].out_frac


def check_exact_range(net: IntSequential, in_max: float) -> float:
    """Worst-case |accumulator| bound; must stay < 2**53 for the float64 trick to be exact."""
    bound, worst = in_max, 0.0
    for m in net:
        if isinstance(m, QConv2d):
            fan_in = m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]
            acc = bound * m.w_max * fan_in + 2 ** (m.acc_frac + 16)
            worst = max(worst, acc)
            bound = 2 ** m.a_bits - 1 if m.act else acc
    assert worst < 2 ** 53, f"accumulator bound {worst:.3g} >= 2**53"
    return worst

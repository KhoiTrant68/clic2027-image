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

import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

EXACT_MODE = {"f32": False, "i8": False}  # set by exact_f32_context() / exact_i8_context()
I8_IN_BITS = 16  # every forward_int input is < 2**16 in magnitude (a_bits = 16 activations, |y_int| <= 2047 * 16,
                 # |z| <= 2**15): three signed 7-bit chunks cover it


@contextmanager
def exact_i8_context(enabled: bool = True):
    """Run forward_int on CUDA with int8 GEMMs (torch._int_mm, int32 accumulation) for full convs and float64
    shift-and-add for depthwise convs. Every step is exact integer arithmetic, so the result equals the float64
    path, and nothing depends on global cuDNN / TF32 flags: the synthesis may run concurrently with cuDNN on
    (engine B, plans/GC_v2_architecture_2026-11-23.md 4.5). CPU tensors keep the float64 path."""
    old = EXACT_MODE["i8"]
    EXACT_MODE["i8"] = enabled
    try:
        yield
    finally:
        EXACT_MODE["i8"] = old


@contextmanager
def exact_f32_context(enabled: bool = True):
    """Run forward_int with the exact fp32-split convolution: TF32 off, cuDNN off (no Winograd/FFT)."""
    old = (EXACT_MODE["f32"], torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    EXACT_MODE["f32"] = enabled
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:  # oneDNN (CPU) may also pick Winograd for fp32: use the plain im2col + GEMM path there too
        with torch.backends.cudnn.flags(enabled=False), torch.backends.mkldnn.flags(enabled=False):
            yield
    finally:
        EXACT_MODE["f32"], torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def ste(x_q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Value of x_q, gradient of x."""
    return x + (x_q - x).detach()


class QConv2d(nn.Conv2d):
    """Integer conv. With n_cond > 0 it also has a per-condition bias (e.g. one per rate level),
    selected by `cond` (LongTensor (B,)); the bias stays an integer, so decoding stays exact."""

    def __init__(self, cin, cout, k, stride=1, *, in_frac: int, out_frac: int | None = None, act: str | None = "relu",
                 w_frac: int = 7, w_max: int = 127, a_bits: int = 16, groups: int = 1, n_cond: int = 0):
        super().__init__(cin, cout, k, stride, padding=k // 2, groups=groups, bias=True)
        self.in_frac, self.w_frac, self.w_max, self.a_bits, self.act = in_frac, w_frac, w_max, a_bits, act
        self.out_frac = out_frac if act else w_frac + in_frac
        assert act in (None, "relu") and self.w_frac + self.in_frac >= self.out_frac
        self.cond_bias = nn.Parameter(torch.zeros(n_cond, cout)) if n_cond else None

    @property
    def acc_frac(self) -> int:
        return self.w_frac + self.in_frac

    def _wq(self):
        return torch.clamp(torch.round(self.weight * 2 ** self.w_frac), -self.w_max, self.w_max)

    def _bias(self, cond, batch):
        b = self.bias.expand(batch, -1)
        if self.cond_bias is not None:
            assert cond is not None, "this layer needs a condition index"
            b = b + self.cond_bias[cond.to(self.cond_bias.device)]
        return b  # (B, cout), float

    def forward(self, x, cond=None):  # float in/out, quantisers with STE
        w = ste(self._wq() / 2 ** self.w_frac, self.weight)
        b = self._bias(cond, x.shape[0])
        b = ste(torch.round(b * 2 ** self.acc_frac) / 2 ** self.acc_frac, b)
        y = F.conv2d(x, w, None, self.stride, self.padding, 1, self.groups) + b[:, :, None, None]
        if self.act is None:
            return y
        # clamp with its true gradient (zero outside the range), then floor with a straight-through gradient;
        # the values are identical to forward_int's clamp(floor(.)) since floor is monotone
        y = torch.clamp(F.relu(y), max=(2 ** self.a_bits - 1) / 2 ** self.out_frac)
        q = torch.floor(y * 2 ** self.out_frac) / 2 ** self.out_frac
        return ste(q, y)

    def _conv_exact_f32(self, x_int: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Exact integer conv with fp32 arithmetic: |x| is split into chunks of s bits so that every partial sum of
        a chunk conv is an integer below 2**24 (exact in fp32 whatever the summation order); chunk results are
        recombined in float64. Same result as the float64 conv, at fp32 speed (FP64 is ~1/64 rate on an L4).
        Needs TF32 off and no Winograd/FFT (cuDNN disabled), see `exact_f32_context`."""
        fan_in = self.in_channels // self.groups * self.kernel_size[0] * self.kernel_size[1]
        s = int(math.floor(math.log2((2 ** 24 - 1) / (self.w_max * fan_in))))
        assert s >= 1, "fan-in too large for the fp32 split"
        sign = torch.sign(x_int)
        mag = x_int.abs()
        top = float(mag.max()) if mag.numel() else 0.0
        n = max(1, math.ceil(math.log2(top + 1) / s))
        wf = w.float()
        acc = None
        for i in range(n):
            chunk = (torch.floor(mag / 2 ** (s * i)) % 2 ** s) * sign
            part = F.conv2d(chunk.float(), wf, None, self.stride, self.padding, 1, self.groups).double() * 2 ** (s * i)
            acc = part if acc is None else acc + part
        return acc

    def _i8_ok(self, x_int: torch.Tensor) -> bool:
        if not (EXACT_MODE["i8"] and x_int.is_cuda and self.stride == (1, 1)):
            return False
        return self.groups == 1 or self.groups == self.in_channels == self.out_channels

    def _w8(self, dev):
        """(K8, N8) int8 GEMM weight (zero-padded to multiples of 8), cached per device and weight version."""
        key = (str(dev), self.weight._version, self.weight.data_ptr())
        if getattr(self, "_w8_key", None) != key:
            cout = self.out_channels
            wq = self._wq().reshape(cout, -1).t()  # (Cin*kh*kw, Cout), cross-correlation order = unfold order
            K, k8, n8 = wq.shape[0], -(-wq.shape[0] // 8) * 8, -(-cout // 8) * 8
            w8 = torch.zeros(n8, k8, dtype=torch.int8, device=dev)  # stored (N, K) row-major: cuBLASLt wants
            w8[:cout, :K] = wq.t().to(dev).to(torch.int8)           # the second int8 operand column-major
            torch.cuda.current_stream(w8.device).synchronize()      # ready before another thread's stream reads it
            self._w8_cache, self._w8_key = w8.t(), key
        return self._w8_cache

    def _conv_exact_i8(self, x_int: torch.Tensor) -> torch.Tensor:
        dev = x_int.device
        B, cin, H, W = x_int.shape
        kh, kw = self.kernel_size
        ph, pw = self.padding
        if self.groups != 1:  # depthwise: float64 shift-and-add (exact: |w x| sums stay far below 2**53)
            w = self._wq().to(dev, torch.float64)
            xp = F.pad(x_int, (pw, pw, ph, ph))
            acc = torch.zeros(B, cin, H, W, dtype=torch.float64, device=dev)
            for dy in range(kh):
                for dx in range(kw):
                    acc += w[:, 0, dy, dx].view(1, -1, 1, 1) * xp[:, :, dy:dy + H, dx:dx + W]
            return acc
        if B * H * W * 3 <= 16:  # too small for _int_mm (m > 16): float64 GEMM, exact and flag-independent too
            A = F.unfold(x_int, (kh, kw), padding=(ph, pw))  # (B, K, HW)
            w = self._wq().reshape(self.out_channels, -1).to(dev, torch.float64)
            return torch.matmul(w, A).reshape(B, self.out_channels, H, W)
        n = -(-I8_IN_BITS // 7)
        sign, mag = torch.sign(x_int), x_int.abs()
        X = torch.stack([torch.remainder(torch.floor(mag / 2 ** (7 * i)), 128) * sign for i in range(n)], 0)
        X = X.reshape(n * B, cin, H, W)
        if kh == kw == 1:
            A = X.permute(0, 2, 3, 1).reshape(n * B * H * W, cin)
        else:  # im2col in fp16: chunk values are integers in [-127, 127], exact in fp16
            A = F.unfold(X.half(), (kh, kw), padding=(ph, pw)).transpose(1, 2).reshape(n * B * H * W, cin * kh * kw)
        w8 = self._w8(dev)
        m = A.shape[0]
        A8 = torch.zeros(-(-m // 8) * 8, w8.shape[0], dtype=torch.int8, device=dev)
        A8[:m, :A.shape[1]] = A.to(torch.int8)
        R = torch._int_mm(A8, w8)[:m, :self.out_channels].double().reshape(n, B, H, W, self.out_channels)
        acc = R[0]
        for i in range(1, n):
            acc = acc + R[i] * 2 ** (7 * i)
        return acc.permute(0, 3, 1, 2)

    @torch.no_grad()
    def forward_int(self, x_int: torch.Tensor, cond=None) -> torch.Tensor:  # float64 tensors holding integers
        dev = x_int.device
        b = torch.round(self._bias(cond, x_int.shape[0]).double() * 2 ** self.acc_frac).to(dev)
        if self._i8_ok(x_int):
            acc = self._conv_exact_i8(x_int.to(torch.float64)) + b[:, :, None, None]
            if self.act is None:
                return acc
            shift = self.acc_frac - self.out_frac
            return torch.clamp(torch.floor(F.relu(acc) / 2 ** shift), 0, 2 ** self.a_bits - 1)
        w = self._wq().to(torch.float64).to(dev)
        if EXACT_MODE["f32"]:
            acc = self._conv_exact_f32(x_int.to(torch.float64), w) + b[:, :, None, None]
        else:
            acc = F.conv2d(x_int.to(torch.float64), w, None, self.stride, self.padding, 1, self.groups) + b[:, :, None, None]
        if self.act is None:
            return acc
        shift = self.acc_frac - self.out_frac
        return torch.clamp(torch.floor(F.relu(acc) / 2 ** shift), 0, 2 ** self.a_bits - 1)


class Up2(nn.Module):
    """Nearest-neighbour x2 upsampling (exact on integers)."""

    def forward(self, x, cond=None):
        return F.interpolate(x, scale_factor=2, mode="nearest")

    forward_int = forward


class IntSequential(nn.Sequential):
    """Sequential that passes an optional condition index to every layer."""

    def forward(self, x, cond=None):
        for m in self:
            x = m(x, cond)
        return x

    def forward_int(self, x_int: torch.Tensor, cond=None) -> torch.Tensor:
        for m in self:
            x_int = m.forward_int(x_int, cond)
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

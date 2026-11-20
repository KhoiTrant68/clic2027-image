"""Training losses: rate control, distortion (MSE, LPIPS, Wasserstein distortion), reward distance."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def rate_lambda(bpp: torch.Tensor, level: torch.Tensor, targets, hi: float, lo: float) -> torch.Tensor:
    """HiFiC-style rate control per sample: a large weight above the level's target bpp, a small one below."""
    t = torch.tensor(targets, device=bpp.device, dtype=bpp.dtype)[level]
    return torch.where(bpp > t, torch.full_like(bpp, hi), torch.full_like(bpp, lo))


class LPIPS(nn.Module):
    """Wrapper around the `lpips` package; inputs in [0, 1], returns (B,)."""

    def __init__(self, net="alex"):
        super().__init__()
        import lpips
        self.m = lpips.LPIPS(net=net, verbose=False).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)

    def forward(self, a, b):
        return self.m(a * 2 - 1, b * 2 - 1).flatten(1).mean(1)


class WassersteinLoss(nn.Module):
    """Differentiable Wasserstein distortion (Qiu, Wagner, Balle, Theis) on VGG16 features with a constant
    log2 sigma, after experiments/qhat/wasserstein.py (BSD-3, Orange). Returns (B,)."""

    LAYERS = (3, 8, 15, 22)

    def __init__(self, log2_sigma=4.0, levels=5):
        super().__init__()
        import torchvision
        from torchvision.models.vgg import VGG16_Weights
        vgg = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[: max(self.LAYERS) + 1]
        self.net = vgg.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        k = torch.tensor([0.25, 0.5, 0.25])
        self.register_buffer("k", torch.outer(k, k).view(1, 1, 3, 3))
        self.ls, self.levels = log2_sigma, levels

    def _feats(self, x):
        out = []
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i in self.LAYERS:
                out.append(x)
        return out

    def _lp(self, x, stride=1):
        B, C, H, W = x.shape
        return F.conv2d(x.reshape(B * C, 1, H, W), self.k, stride=stride, padding=1).reshape(B, C, *(
            (H + stride - 1) // stride, (W + stride - 1) // stride))

    def forward(self, a, b):
        tot = 0.0
        for fa, fb in zip(self._feats(a), self._feats(b)):
            maps = [((fa - fb) ** 2).mean(1, keepdim=True)]
            sa, sb = fa, fb
            qa, qb = fa * fa, fb * fb
            for _ in range(self.levels):
                ma, mb, pa, pb = self._lp(sa), self._lp(sb), self._lp(qa), self._lp(qb)
                va, vb = pa - ma ** 2, pb - mb ** 2
                d = (ma - mb) ** 2 + (va.clamp_min(5e-7).sqrt() - vb.clamp_min(5e-7).sqrt()) ** 2
                maps.append(d.mean(1, keepdim=True))
                sa, sb, qa, qb = ma[..., ::2, ::2], mb[..., ::2, ::2], pa[..., ::2, ::2], pb[..., ::2, ::2]
            per = 0.0
            for i, m in enumerate(maps):
                w = max(0.0, 1 - abs(self.ls - i))
                if w:
                    per = per + w * m.flatten(1).mean(1)
            tot = tot + per
        return tot


def psnr_t(a, b):
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1 / mse)

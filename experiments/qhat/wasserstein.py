"""Wasserstein Distortion (WD) as an image-quality feature for Q-hat.

WD (Qiu, Wagner, Balle, Theis; used for compression in Balle et al., "Good, Cheap, and Fast: Overfitted Image
Compression with Wasserstein Distortion", CVPR 2025) compares local feature statistics (mean and standard deviation
pooled over a window of size sigma) instead of the features themselves, so a texture that is re-synthesised rather
than copied costs little. Balle et al. report > 94% Pearson correlation of WD with CLIC Elo scores.

Adapted from Cool-chic's coolchic/training/metrics/wasserstein.py (Orange, BSD-3-Clause), itself inspired by
google/codex. Differences: no caching of the reference features between calls (Cool-chic encodes one image, we
score many), no gradients, a constant log2(sigma) per call, and tiling for 2K images.

Copyright (c) 2023-2025 Orange (original code); BSD-3-Clause.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

LAYERS = (3, 8, 15, 22)  # VGG16 relu1_2, relu2_2, relu3_3, relu4_3 (right before the max-pools)
NUM_LEVELS = 5


class WassersteinDistortion:
    """wd(o, x) for HxWx3 uint8 arrays -> {log2_sigma: WD}; lower is better. Same VGG16 input convention as
    Cool-chic: images in [0, 1], no ImageNet normalisation."""

    def __init__(self, dev):
        import torchvision
        from torchvision.models.vgg import VGG16_Weights
        vgg = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[: max(LAYERS) + 1]
        self.net = vgg.to(dev).eval()
        self.dev = dev
        k = torch.tensor([0.25, 0.5, 0.25])
        self.kernel = torch.outer(k, k).view(1, 1, 3, 3).to(dev)

    def _features(self, x):
        out = []
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i in LAYERS:
                out.append(x[0][:, None])  # (C, 1, h, w): every channel is filtered on its own
        return out

    def _lowpass(self, x, stride):
        return F.conv2d(x, self.kernel, stride=stride, padding=1)

    def _stats(self, f):
        sq = f * f
        means, variances = [], []
        for _ in range(NUM_LEVELS):
            m = self._lowpass(f, 1)
            p = self._lowpass(sq, 1)
            means.append(m)
            variances.append(p - m * m)
            f, sq = m[..., ::2, ::2], p[..., ::2, ::2]
        return means, variances

    def _wd_one(self, fa, fb, log2_sigma):
        ma, va = self._stats(fa)
        mb, vb = self._stats(fb)
        maps = [(fa - fb) ** 2]
        for a_m, b_m, a_v, b_v in zip(ma, mb, va, vb):
            maps.append((a_m - b_m) ** 2 + (a_v.clamp_min(5e-7).sqrt() - b_v.clamp_min(5e-7).sqrt()) ** 2)
        ls = torch.full((1, 1, *fa.shape[-2:]), float(log2_sigma), device=fa.device)
        d = 0.0
        for i, m in enumerate(maps):
            d += float(torch.mean(F.relu(1 - (ls - i).abs()) * m))
            if i > 0:
                ls = self._lowpass(ls, 2)
        return d

    @torch.no_grad()
    def tile(self, a, b, log2_sigmas):
        """a, b: 1x3xHxW in [0, 1]. Returns {log2_sigma: WD} summed over the VGG layers (as in Cool-chic)."""
        fa, fb = self._features(a), self._features(b)
        return {s: sum(self._wd_one(x, y, s) for x, y in zip(fa, fb)) for s in log2_sigmas}

    def __call__(self, o, x, log2_sigmas=(3, 5), tile=512):
        """Area-weighted mean over tile x tile crops (tiles with a side < 64 are skipped), like LPIPS / DISTS."""
        from clic27.eval import metrics as M
        H, W = o.shape[:2]
        tot = {s: 0.0 for s in log2_sigmas}
        wsum = 0.0
        for y in range(0, H, tile):
            for xx in range(0, W, tile):
                pa, pb = x[y:y + tile, xx:xx + tile], o[y:y + tile, xx:xx + tile]
                if min(pa.shape[:2]) < 64:
                    continue
                n = pa.shape[0] * pa.shape[1]
                for s, v in self.tile(M.to_tensor(pa, self.dev), M.to_tensor(pb, self.dev), log2_sigmas).items():
                    tot[s] += v * n
                wsum += n
        return {s: v / wsum for s, v in tot.items()}

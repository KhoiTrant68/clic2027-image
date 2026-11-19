"""S1: variable-rate hyperprior codec on DC-AE f32c32 latents, with subtractive dither in the bitstream.

    latent (B, 32, h, w), already multiplied by the DC-AE scaling factor
      g_a ->  y (B, M, h/s, w/s)             s = y_stride (1 or 2)
      y_r = y * gain[r]                      quantisation step 1/gain[r] per channel (rate level r)
      h_a(y_r) -> z (B, Z, h/4s, w/4s)       z_hat = round(z), DiscretePrior
      h_s(z_hat; r) [integer net] -> (scale index, mu)   (sigma, mu) during training
      y_r_hat = q - u,  q = round(y_r + u)   shared dither u (Prop. 1): error ~ U(-1/2, 1/2), independent of y
      y_hat = y_r_hat / gain[r];  g_s -> latent_hat  (trained with MSE, so latent_hat ~ E[latent | bitstream])

Everything that decides a CDF (h_s, tables, dither, centre) is integer/bit-exact; g_a, h_a, g_s are float.
`compress` / `decompress` produce and parse a self-contained byte string per image.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..entropy import rans
from ..entropy.intnet import IntSequential, QConv2d, Up2
from ..entropy.models import DiscretePrior, DitheredGaussian, ScaleMeanHead, gaussian_bits

MAGIC = 0x52  # 'R'
VERSION = 1


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, 1, 1)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.c2(self.act(self.c1(x)))


class LatentCodec(nn.Module):
    def __init__(self, latent_ch: int = 32, N: int = 192, M: int = 128, Z: int = 96, n_rates: int = 8,
                 y_stride: int = 2, n_offsets: int = 16, mu_frac: int = 4, gain_range=(0.25, 8.0)):
        super().__init__()
        assert y_stride in (1, 2)
        self.config = dict(latent_ch=latent_ch, N=N, M=M, Z=Z, n_rates=n_rates, y_stride=y_stride,
                           n_offsets=n_offsets, mu_frac=mu_frac, gain_range=list(gain_range))
        self.y_stride, self.mu_frac, self.n_rates = y_stride, mu_frac, n_rates
        self.multiple = 4 * y_stride  # latent H, W are padded to this

        down = [nn.Conv2d(N, N, 3, 2, 1)] if y_stride == 2 else []
        up = [nn.Conv2d(N, 4 * N, 3, 1, 1), nn.PixelShuffle(2)] if y_stride == 2 else []
        self.g_a = nn.Sequential(nn.Conv2d(latent_ch, N, 3, 1, 1), ResBlock(N), ResBlock(N), *down,
                                 ResBlock(N), ResBlock(N), nn.Conv2d(N, M, 3, 1, 1))
        self.g_s = nn.Sequential(nn.Conv2d(M, N, 3, 1, 1), ResBlock(N), ResBlock(N), *up,
                                 ResBlock(N), ResBlock(N), nn.Conv2d(N, latent_ch, 3, 1, 1))
        self.h_a = nn.Sequential(nn.Conv2d(M, N, 3, 1, 1), nn.LeakyReLU(0.1), nn.Conv2d(N, N, 5, 2, 2),
                                 nn.LeakyReLU(0.1), nn.Conv2d(N, Z, 5, 2, 2))
        self.h_s = IntSequential(
            Up2(), QConv2d(Z, N, 5, in_frac=0, out_frac=7, n_cond=n_rates),
            Up2(), QConv2d(N, N, 5, in_frac=7, out_frac=7, n_cond=n_rates),
            QConv2d(N, 2 * M, 3, in_frac=7, act=None, n_cond=n_rates),
        )
        self.head = ScaleMeanHead(self.h_s[-1].acc_frac, mu_frac)
        self.z_prior = DiscretePrior(Z)
        self.y_model = DitheredGaussian(n_offsets)
        g = torch.exp(torch.linspace(math.log(gain_range[0]), math.log(gain_range[1]), n_rates))
        self.log_gain = nn.Parameter(torch.log(g)[:, None].repeat(1, M))  # (n_rates, M)

    # ------------------------------------------------------------------ helpers
    def _pad(self, lat):
        h, w = lat.shape[-2:]
        m = self.multiple
        ph, pw = (-h) % m, (-w) % m
        if ph or pw:
            lat = F.pad(lat, (0, pw, 0, ph), mode="reflect" if min(h, w) > max(ph, pw) else "replicate")
        return lat, (h, w)

    def gain(self, r: torch.Tensor):
        return torch.exp(self.log_gain[r])[:, :, None, None]

    # ------------------------------------------------------------------ training / evaluation forward
    def forward(self, lat: torch.Tensor, r: torch.Tensor) -> dict:
        """lat: (B, 32, h, w) scaled latent; r: (B,) rate levels. Bits are exact code lengths of q given u
        (up to table quantisation), so bpp is comparable with real bitstreams."""
        x, (h, w) = self._pad(lat)
        g = self.gain(r)
        y_r = self.g_a(x) * g
        z_hat, bits_z = self.z_prior(self.h_a(y_r))
        sigma, mu = self.head(self.h_s(z_hat, r))
        y_r_hat = DitheredGaussian.quantize(y_r)
        bits_y = gaussian_bits(y_r_hat, sigma, mu)
        lat_hat = self.g_s(y_r_hat / g)[..., :h, :w]
        return dict(lat_hat=lat_hat, bits_y=bits_y.flatten(1).sum(1), bits_z=bits_z.flatten(1).sum(1),
                    y_r=y_r, sigma=sigma)

    # ------------------------------------------------------------------ bitstream
    @torch.no_grad()
    def compress(self, lat: torch.Tensor, r: int, seed: int = 0) -> bytes:
        """lat: (1, 32, h, w) scaled latent. Returns the full byte string for one image."""
        assert lat.shape[0] == 1
        x, (h, w) = self._pad(lat)
        rr = torch.tensor([r], device=lat.device)
        y_r = self.g_a(x) * self.gain(rr)
        z_int = torch.round(self.h_a(y_r))
        zb = self.z_prior.compress(z_int)
        idx, mu_int = self._coding_params(z_int, r)
        yb = self.y_model.compress(y_r, idx, mu_int, self.mu_frac, seed)
        out = bytearray([MAGIC, VERSION])
        for v in (h, w, r, seed, len(zb)):
            rans.put_varint(int(v), out)
        return bytes(out + zb + yb)

    @torch.no_grad()
    def decompress(self, blob: bytes, device="cpu") -> torch.Tensor:
        assert blob[0] == MAGIC and blob[1] == VERSION, "not a clic27 S1 stream"
        pos, hdr = 2, []
        for _ in range(5):
            v, pos = rans.get_varint(blob, pos)
            hdr.append(v)
        h, w, r, seed, nz = hdr
        m = self.multiple
        H, W = (h + (-h) % m) // self.y_stride, (w + (-w) % m) // self.y_stride
        z_shape = (1, self.config["Z"], H // 4, W // 4)
        z_int = self.z_prior.decompress(blob[pos:pos + nz], z_shape)
        idx, mu_int = self._coding_params(z_int, r)
        y_r_hat = self.y_model.decompress(blob[pos + nz:], idx, mu_int, self.mu_frac, seed).to(device)
        rr = torch.tensor([r], device=device)
        return self.g_s(y_r_hat / self.gain(rr).to(device))[..., :h, :w]

    def _coding_params(self, z_int: torch.Tensor, r: int):
        """Integer path on CPU (bit-exact on every machine)."""
        acc = self.h_s.forward_int(z_int.double().cpu(), torch.tensor([r]))
        return self.head.coding_params(acc)

    @torch.no_grad()
    def prepare_coding(self):
        """Freeze the learned z prior into integer tables. Call before compress/decompress and before saving."""
        self.z_prior.update_tables()
        return self

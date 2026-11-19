"""DC-AE (Deep Compression Autoencoder) in plain torch, ported from diffusers 0.40.0
`models/autoencoders/autoencoder_dc.py`. Only what SANA's f32c32 VAE uses is kept
(ResBlock + EfficientViTBlock, rms_norm, conv downsample, interpolate upsample, tiling).

    ae = DCAE.from_pretrained("Efficient-Large-Model/Sana_1600M_1024px_diffusers", "vae")
    z = ae.encode(x)          # x in [-1, 1], NCHW; z unscaled (multiply by ae.scaling_factor for the DiT)
    x_hat = ae.decode(z)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .common import GLUMBConv, RMSNorm, channel_rms_norm, get_activation, linear_attention
from .weights import load_config, load_state_dict, resolve


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_type: str, act_fn: str):
        super().__init__()
        assert norm_type == "rms_norm"
        self.nonlinearity = get_activation(act_fn)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.norm = RMSNorm(out_channels, eps=1e-5, elementwise_affine=True, bias=True)

    def forward(self, x):
        h = self.conv2(self.nonlinearity(self.conv1(x)))
        return channel_rms_norm(self.norm, h) + x


class MultiscaleProjection(nn.Module):
    """diffusers SanaMultiscaleAttentionProjection."""

    def __init__(self, in_channels: int, num_heads: int, kernel_size: int):
        super().__init__()
        c = 3 * in_channels
        self.proj_in = nn.Conv2d(c, c, kernel_size, padding=kernel_size // 2, groups=c, bias=False)
        self.proj_out = nn.Conv2d(c, c, 1, 1, 0, groups=3 * num_heads, bias=False)

    def forward(self, x):
        return self.proj_out(self.proj_in(x))


class MultiscaleLinearAttention(nn.Module):
    """diffusers SanaMultiscaleLinearAttention + SanaMultiscaleAttnProcessor2_0 (rms_norm, residual)."""

    def __init__(self, channels: int, attention_head_dim: int, kernel_sizes, mult: float = 1.0, eps: float = 1e-15):
        super().__init__()
        self.eps = eps
        self.attention_head_dim = attention_head_dim
        heads = int(channels // attention_head_dim * mult)
        inner = heads * attention_head_dim
        self.to_q = nn.Linear(channels, inner, bias=False)
        self.to_k = nn.Linear(channels, inner, bias=False)
        self.to_v = nn.Linear(channels, inner, bias=False)
        self.to_qkv_multiscale = nn.ModuleList(MultiscaleProjection(inner, heads, k) for k in kernel_sizes)
        self.nonlinearity = nn.ReLU()
        self.to_out = nn.Linear(inner * (1 + len(kernel_sizes)), channels, bias=False)
        self.norm_out = RMSNorm(channels, eps=1e-5, elementwise_affine=True, bias=True)

    def forward(self, x):
        B, _, H, W = x.shape
        use_linear = H * W > self.attention_head_dim
        residual, dtype = x, x.dtype
        h = x.movedim(1, -1)
        h = torch.cat([self.to_q(h), self.to_k(h), self.to_v(h)], dim=3).movedim(-1, 1)
        h = torch.cat([h] + [blk(h) for blk in self.to_qkv_multiscale], dim=1)
        if use_linear:
            h = h.to(torch.float32)
        h = h.reshape(B, -1, 3 * self.attention_head_dim, H * W)
        q, k, v = h.chunk(3, dim=2)
        q, k = self.nonlinearity(q), self.nonlinearity(k)
        if use_linear:
            h = linear_attention(q, k.transpose(-1, -2), v, self.eps).to(dtype)
        else:  # quadratic attention for tiny maps (diffusers apply_quadratic_attention)
            s = torch.matmul(k.transpose(-1, -2), q).to(torch.float32)
            s = s / (torch.sum(s, dim=2, keepdim=True) + self.eps)
            h = torch.matmul(v, s.to(v.dtype))
        h = torch.reshape(h, (B, -1, H, W))
        h = self.to_out(h.movedim(1, -1)).movedim(-1, 1)
        return channel_rms_norm(self.norm_out, h) + residual


class EfficientViTBlock(nn.Module):
    def __init__(self, channels: int, attention_head_dim: int, qkv_multiscales):
        super().__init__()
        self.attn = MultiscaleLinearAttention(channels, attention_head_dim, qkv_multiscales)
        self.conv_out = GLUMBConv(channels, channels, norm_type="rms_norm")

    def forward(self, x):
        return self.conv_out(self.attn(x))


def _per_block(v, n):
    """Config values may be one string for all blocks or a per-block list."""
    return [v] * n if isinstance(v, str) else list(v)


def make_block(kind, channels, attention_head_dim, norm_type, act_fn, qkv_multiscales):
    if kind == "ResBlock":
        return ResBlock(channels, channels, norm_type, act_fn)
    if kind == "EfficientViTBlock":
        return EfficientViTBlock(channels, attention_head_dim, qkv_multiscales)
    raise ValueError(kind)


class DownBlock(nn.Module):
    """DCDownBlock2d with downsample=False (strided conv), i.e. downsample_block_type='Conv'."""

    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True):
        super().__init__()
        self.group_size = in_channels * 4 // out_channels
        self.shortcut = shortcut
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1)

    def forward(self, x):
        h = self.conv(x)
        if self.shortcut:
            y = F.pixel_unshuffle(x, 2).unflatten(1, (-1, self.group_size)).mean(dim=2)
            h = h + y
        return h


class UpBlock(nn.Module):
    """DCUpBlock2d with interpolate=True (nearest), i.e. upsample_block_type='interpolate'."""

    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True):
        super().__init__()
        self.repeats = out_channels * 4 // in_channels
        self.shortcut = shortcut
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        h = self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))
        if self.shortcut:
            y = x.repeat_interleave(self.repeats, dim=1, output_size=x.shape[1] * self.repeats)
            h = h + F.pixel_shuffle(y, 2)
        return h


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ch, layers = cfg["encoder_block_out_channels"], cfg["encoder_layers_per_block"]
        types, ms = _per_block(cfg["encoder_block_types"], len(ch)), cfg["encoder_qkv_multiscales"]
        assert layers[0] > 0 and cfg["downsample_block_type"] == "Conv"
        self.conv_in = nn.Conv2d(cfg["in_channels"], ch[0], 3, 1, 1)
        blocks = []
        for i, (c, n) in enumerate(zip(ch, layers)):
            seq = [make_block(types[i], c, cfg["attention_head_dim"], "rms_norm", "silu", ms[i]) for _ in range(n)]
            if i < len(ch) - 1 and n > 0:
                seq.append(DownBlock(c, ch[i + 1], shortcut=True))
            blocks.append(nn.Sequential(*seq))
        self.down_blocks = nn.ModuleList(blocks)
        self.conv_out = nn.Conv2d(ch[-1], cfg["latent_channels"], 3, 1, 1)
        self.out_group = ch[-1] // cfg["latent_channels"]

    def forward(self, x):
        h = self.conv_in(x)
        for blk in self.down_blocks:
            h = blk(h)
        return self.conv_out(h) + h.unflatten(1, (-1, self.out_group)).mean(dim=2)


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ch, layers = cfg["decoder_block_out_channels"], cfg["decoder_layers_per_block"]
        types, ms = _per_block(cfg["decoder_block_types"], len(ch)), cfg["decoder_qkv_multiscales"]
        norms = _per_block(cfg.get("decoder_norm_types", "rms_norm"), len(ch))
        acts = _per_block(cfg.get("decoder_act_fns", "silu"), len(ch))
        assert layers[0] > 0 and cfg["upsample_block_type"] == "interpolate"
        self.conv_in = nn.Conv2d(cfg["latent_channels"], ch[-1], 3, 1, 1)
        self.in_repeats = ch[-1] // cfg["latent_channels"]
        up = []
        for i, (c, n) in reversed(list(enumerate(zip(ch, layers)))):
            seq = []
            if i < len(ch) - 1 and n > 0:
                seq.append(UpBlock(ch[i + 1], c, shortcut=True))
            seq += [make_block(types[i], c, cfg["attention_head_dim"], norms[i], acts[i], ms[i]) for _ in range(n)]
            up.insert(0, nn.Sequential(*seq))
        self.up_blocks = nn.ModuleList(up)
        self.norm_out = RMSNorm(ch[0], 1e-5, elementwise_affine=True, bias=True)
        self.conv_act = nn.ReLU()
        self.conv_out = nn.Conv2d(ch[0], cfg["in_channels"], 3, 1, 1)

    def forward(self, z):
        rep = z.repeat_interleave(self.in_repeats, dim=1, output_size=z.shape[1] * self.in_repeats)
        h = self.conv_in(z) + rep
        for blk in reversed(self.up_blocks):
            h = blk(h)
        return self.conv_out(self.conv_act(channel_rms_norm(self.norm_out, h)))


class DCAE(nn.Module):
    # diffusers AutoencoderDC tiling defaults (sample space)
    TILE, STRIDE = 512, 448

    def __init__(self, cfg: dict):
        super().__init__()
        self.config = dict(cfg)
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.f = 2 ** (len(cfg["encoder_block_out_channels"]) - 1)
        self.scaling_factor = cfg.get("scaling_factor", 1.0)
        self.use_tiling = False

    @classmethod
    def from_pretrained(cls, repo_or_dir: str, subfolder: str | None = "vae", dtype=torch.float32):
        d = resolve(repo_or_dir, subfolder)
        with torch.device("meta"):  # no random init / no fp32 copy of the weights in RAM
            m = cls(load_config(d))
        m.load_state_dict(load_state_dict(d), strict=True, assign=True)
        return m.to(dtype).eval()

    def enable_tiling(self, on: bool = True):
        self.use_tiling = on
        return self

    # -- encode / decode (same tiling rule and blending as diffusers, so ceilings match) --
    def encode(self, x):
        if self.use_tiling and (x.shape[-1] > self.TILE or x.shape[-2] > self.TILE):
            return self._tiled_encode(x)
        return self.encoder(x)

    def decode(self, z):
        t = self.TILE // self.f
        if self.use_tiling and (z.shape[-1] > t or z.shape[-2] > t):
            return self._tiled_decode(z)
        return self.decoder(z)

    @staticmethod
    def _blend_v(a, b, n):
        n = min(a.shape[2], b.shape[2], n)
        for y in range(n):
            b[:, :, y, :] = a[:, :, -n + y, :] * (1 - y / n) + b[:, :, y, :] * (y / n)
        return b

    @staticmethod
    def _blend_h(a, b, n):
        n = min(a.shape[3], b.shape[3], n)
        for x in range(n):
            b[:, :, :, x] = a[:, :, :, -n + x] * (1 - x / n) + b[:, :, :, x] * (x / n)
        return b

    def _tiled_encode(self, x):
        f, T, S = self.f, self.TILE, self.STRIDE
        lh, lw = x.shape[2] // f, x.shape[3] // f
        tl, sl = T // f, S // f
        rows = []
        for i in range(0, x.shape[2], S):
            row = []
            for j in range(0, x.shape[3], S):
                tile = x[:, :, i:i + T, j:j + T]
                if tile.shape[2] % f or tile.shape[3] % f:
                    tile = F.pad(tile, (0, (f - tile.shape[3]) % f, 0, (f - tile.shape[2]) % f))
                row.append(self.encoder(tile))
            rows.append(row)
        return self._stitch(rows, tl - sl, sl)[:, :, :lh, :lw]

    def _tiled_decode(self, z):
        f, T, S = self.f, self.TILE, self.STRIDE
        tl, sl = T // f, S // f
        rows = [[self.decoder(z[:, :, i:i + tl, j:j + tl]) for j in range(0, z.shape[3], sl)]
                for i in range(0, z.shape[2], sl)]
        return self._stitch(rows, T - S, S)

    def _stitch(self, rows, blend, stride):
        out_rows = []
        for i, row in enumerate(rows):
            out = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend_v(rows[i - 1][j], tile, blend)
                if j > 0:
                    tile = self._blend_h(row[j - 1], tile, blend)
                out.append(tile[:, :, :stride, :stride])
            out_rows.append(torch.cat(out, dim=3))
        return torch.cat(out_rows, dim=2)

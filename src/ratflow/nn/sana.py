"""SANA DiT (SanaTransformer2DModel) in plain torch, ported from diffusers 0.40.0
`models/transformers/sana_transformer.py`, for the configs of Sana_600M / Sana_1600M
(patch 1, no positional embedding, no guidance embedding, no qk-norm).

    dit = SanaDiT.from_pretrained("Efficient-Large-Model/Sana_1600M_1024px_diffusers", "transformer")
    v = dit(x, t, context)        # x: (B, 32, h, w) latent; t: (B,) in [0, 1000]; context: (B, L, 2304)

`expand_in_channels` adds zero-initialised input channels (e.g. to concatenate the decoded ŷ).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from .common import GLUMBConv, RMSNorm, linear_attention
from .weights import load_config, load_state_dict, resolve


def timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: int = 10000) -> torch.Tensor:
    """diffusers get_timestep_embedding with flip_sin_to_cos=True, downscale_freq_shift=0."""
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    emb = t[:, None].float() * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return torch.cat([emb[:, half:], emb[:, :half]], dim=-1)


class MLP2(nn.Module):
    """linear_1 -> act -> linear_2 (TimestepEmbedding / PixArtAlphaTextProjection)."""

    def __init__(self, d_in, d_hidden, act):
        super().__init__()
        self.linear_1 = nn.Linear(d_in, d_hidden)
        self.act = act
        self.linear_2 = nn.Linear(d_hidden, d_hidden)

    def forward(self, x):
        return self.linear_2(self.act(self.linear_1(x)))


class TimestepEmb(nn.Module):  # PixArtAlphaCombinedTimestepSizeEmbeddings without size conditions
    def __init__(self, dim):
        super().__init__()
        self.timestep_embedder = MLP2(256, dim, nn.SiLU())

    def forward(self, t, dtype):
        return self.timestep_embedder(timestep_embedding(t).to(dtype))


class AdaLNSingle(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.emb = TimestepEmb(dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 6 * dim)

    def forward(self, t, dtype):
        e = self.emb(t, dtype)
        return self.linear(self.silu(e)), e


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, dim, patch):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """diffusers Attention with SanaLinearAttnProcessor2_0 (self) or SanaAttnProcessor2_0 (cross)."""

    def __init__(self, dim, heads, head_dim, bias, cross_dim=None, linear=False, out_bias=True):
        super().__init__()
        inner = heads * head_dim
        self.heads, self.linear = heads, linear
        self.to_q = nn.Linear(dim, inner, bias=bias)
        self.to_k = nn.Linear(cross_dim or dim, inner, bias=bias)
        self.to_v = nn.Linear(cross_dim or dim, inner, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(inner, dim, bias=out_bias), nn.Dropout(0.0)])

    def forward(self, x, context=None, mask=None):
        dtype = x.dtype
        ctx = x if context is None else context
        q, k, v = self.to_q(x), self.to_k(ctx), self.to_v(ctx)
        if self.linear:  # SanaLinearAttnProcessor2_0
            q = F.relu(q.transpose(1, 2).unflatten(1, (self.heads, -1)))
            k = F.relu(k.transpose(1, 2).unflatten(1, (self.heads, -1)).transpose(2, 3))
            v = v.transpose(1, 2).unflatten(1, (self.heads, -1))
            h = linear_attention(q.float(), k.float(), v.float())
            h = h.flatten(1, 2).transpose(1, 2).to(dtype)
            h = self.to_out[1](self.to_out[0](h))
            return h.clip(-65504, 65504) if dtype == torch.float16 else h
        B = x.shape[0]  # SanaAttnProcessor2_0
        hd = k.shape[-1] // self.heads
        q = q.view(B, -1, self.heads, hd).transpose(1, 2)
        k = k.view(B, -1, self.heads, hd).transpose(1, 2)
        v = v.view(B, -1, self.heads, hd).transpose(1, 2)
        if mask is not None:  # (B, L) keep-mask -> additive bias (B, 1, 1, L)
            mask = ((1 - mask.to(q.dtype)) * -10000.0)[:, None, None, :]
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        h = h.transpose(1, 2).reshape(B, -1, self.heads * hd).to(q.dtype)
        return self.to_out[1](self.to_out[0](h))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["num_attention_heads"] * c["attention_head_dim"]
        eps = c["norm_eps"]
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=eps)
        self.attn1 = Attention(d, c["num_attention_heads"], c["attention_head_dim"], c["attention_bias"], linear=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=c["norm_elementwise_affine"], eps=eps)
        self.attn2 = Attention(d, c["num_cross_attention_heads"], c["cross_attention_head_dim"], True,
                               cross_dim=c["cross_attention_dim"])
        self.ff = GLUMBConv(d, d, c["mlp_ratio"], norm_type=None, residual_connection=False)
        self.scale_shift_table = nn.Parameter(torch.randn(6, d) / d ** 0.5)

    def forward(self, x, context, context_mask, t6, H, W):
        B = x.shape[0]
        sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = (self.scale_shift_table[None] + t6.reshape(B, 6, -1)).chunk(6, dim=1)
        n = (self.norm1(x) * (1 + sc_msa) + sh_msa).to(x.dtype)
        x = x + g_msa * self.attn1(n)
        x = self.attn2(x, context, context_mask) + x
        n = self.norm2(x) * (1 + sc_mlp) + sh_mlp
        n = n.unflatten(1, (H, W)).permute(0, 3, 1, 2)
        ff = self.ff(n).flatten(2, 3).permute(0, 2, 1)
        return x + g_mlp * ff


class SanaDiT(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        c = dict(cfg)
        assert not c.get("guidance_embeds") and not c.get("qk_norm") and c.get("interpolation_scale") is None
        self.config = c
        d = c["num_attention_heads"] * c["attention_head_dim"]
        self.patch = c["patch_size"]
        self.out_channels = c.get("out_channels") or c["in_channels"]
        self.patch_embed = PatchEmbed(c["in_channels"], d, self.patch)
        self.time_embed = AdaLNSingle(d)
        self.caption_projection = MLP2(c["caption_channels"], d, nn.GELU(approximate="tanh"))
        self.caption_norm = RMSNorm(d, eps=1e-5, elementwise_affine=True)
        self.transformer_blocks = nn.ModuleList(Block(c) for _ in range(c["num_layers"]))
        self.scale_shift_table = nn.Parameter(torch.randn(2, d) / d ** 0.5)
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)  # SanaModulatedNorm.norm
        self.proj_out = nn.Linear(d, self.patch * self.patch * self.out_channels)
        self.gradient_checkpointing = False

    @classmethod
    def from_pretrained(cls, repo_or_dir: str, subfolder: str | None = "transformer", dtype=torch.float32):
        d = resolve(repo_or_dir, subfolder)
        m = cls(load_config(d))
        sd = load_state_dict(d)
        # diffusers keeps the modulated norm under norm_out.norm (no params since elementwise_affine=False)
        m.load_state_dict(sd, strict=True)
        return m.to(dtype).eval()

    def forward(self, x, t, context, context_mask=None):
        B, _, h, w = x.shape
        p = self.patch
        H, W = h // p, w // p
        x = self.patch_embed(x)
        t6, t_emb = self.time_embed(t, x.dtype)
        ctx = self.caption_norm(self.caption_projection(context).view(B, -1, x.shape[-1]))
        for blk in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(blk, x, ctx, context_mask, t6, H, W, use_reentrant=False)
            else:
                x = blk(x, ctx, context_mask, t6, H, W)
        shift, scale = (self.scale_shift_table[None] + t_emb[:, None].to(self.scale_shift_table.device)).chunk(2, dim=1)
        x = self.proj_out(self.norm_out(x) * (1 + scale) + shift)
        x = x.reshape(B, H, W, p, p, -1).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(B, -1, H * p, W * p)


def expand_in_channels(dit: SanaDiT, extra: int) -> SanaDiT:
    """Add `extra` zero-initialised input channels to the patch embedding (the pretrained path is unchanged)."""
    old = dit.patch_embed.proj
    new = nn.Conv2d(old.in_channels + extra, old.out_channels, old.kernel_size, old.stride,
                    device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :old.in_channels] = old.weight
        new.bias.copy_(old.bias)
    dit.patch_embed.proj = new
    dit.config["in_channels"] = old.in_channels + extra
    return dit

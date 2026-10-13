"""Shared pieces for distilling DiT-IC's DC-AE decoder into a lighter one (2026-10-11).

Teacher: the LoRA-merged DC-AE decoder of one DiT-IC checkpoint (q2/q3/q4 each have their own). Student: the same
decoder with half the channels in its three high-resolution ResBlock stages (512/256/128 -> 256/128/64, ~75% of the
FLOPs); conv_in and the three low-resolution EfficientViT stages (1/32 .. 1/8) are copied from the teacher and frozen,
so teacher and student share that prefix and only the suffix is trained.

Run on the server with the DiT-IC checkout importable (DITIC_DIR).
"""
from __future__ import annotations

import math
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DITIC_DIR = Path(os.environ.get("DITIC_DIR", "/mnt/home/guess03/clic/work_new/DiT-IC"))  # DiT-IC checkout
BASE = "Efficient-Large-Model/Sana_600M_1024px_diffusers"
HF = "JunqiShi/DiT-IC"
STUDENT_CHANNELS = (64, 128, 256, 512, 1024, 1024)
TRAINED = (0, 1, 2)  # up_blocks indices of the student's own (high-resolution) stages; 3, 4, 5 come from the teacher


def load_ditic(q: str, device: str = "cuda"):
    """DiT-IC Codec for checkpoint q (merged LoRA), with its training-only modules stubbed out."""
    sys.path.insert(0, str(DITIC_DIR))
    import yaml
    from huggingface_hub import hf_hub_download
    import models  # noqa: F401  DiT-IC's package

    class Unused(nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

    for mod, attr, val in (("models.losses", "LPIPSWithDiscriminator", Unused),
                           ("models.foundation_models", "aux_foundation_model", None)):
        sys.modules[mod] = types.ModuleType(mod)
        setattr(sys.modules[mod], attr, val)
    import diffusers  # noqa: F401  before the stub below: diffusers probes for peft when imported
    try:
        import peft  # noqa: F401  only used for LoRA training; the merged checkpoints never call it
    except ImportError:
        sys.modules["peft"] = types.ModuleType("peft")
        sys.modules["peft"].get_peft_model = sys.modules["peft"].LoraConfig = None
    from models.DiT_IC import Codec
    cfg = yaml.safe_load(open(DITIC_DIR / "configs" / "inference_merge.yaml"))["model"]["params"]
    cfg.update(dit_path=BASE, use_ema=True, use_merge=True, training=False)
    net = Codec(device, "Codec", **dict(cfg, codec_path=hf_hub_download(HF, f"{q}_merge_ema.pt")))
    net = net.to(device).eval()
    net.codec.update(force=True)
    return net


@torch.no_grad()
def ditic_latent(net, x_u8, dit_dtype=torch.bfloat16):
    """HxWx3 uint8 -> (DC-AE latent z cut to the image, i.e. what the decoder gets, (1, 32, ceil(H/32), ceil(W/32));
    bitstream bytes). Encoder side fp32 and tiled; DiT in dit_dtype as deployed."""
    H, W = x_u8.shape[:2]
    t = torch.from_numpy(x_u8.copy()).permute(2, 0, 1)[None].float().div(127.5).sub(1).cuda()
    t = F.pad(t, (0, (-W) % 256, 0, (-H) % 256), mode="reflect")
    net.vae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024,
                          tile_sample_stride_height=896, tile_sample_stride_width=896)
    o = net.compress(t)
    nbytes = sum(len(s[0]) for s in o["strings"])
    mean, logs, res, prompt = net.codec.decompress(o["strings"], o["shape"])
    scale = torch.exp(0.5 * torch.clamp(logs, -30.0, 20.0))
    enc = net.prompter.to(dit_dtype)(prompt.to(dit_dtype))
    dit = net.DiT.to(dit_dtype)
    ts = net.sched.base_scheduler.timesteps.expand(1) * dit.config.timestep_scale
    pred = dit(mean.to(dit_dtype), encoder_hidden_states=enc, timestep=ts.to(mean.device), return_dict=False)[0]
    z = (net.sched.step(pred.float(), scale, net.timesteps, mean, return_dict=True) + res) / net.vae.config.scaling_factor
    return z[:, :, :math.ceil(H / 32), :math.ceil(W / 32)], nbytes


def build_student(teacher_decoder, channels=STUDENT_CHANNELS):
    """Student decoder: teacher's config with `channels` (the last three must be the teacher's); conv_in and
    up_blocks 3..5 copied from the teacher."""
    from diffusers.models.autoencoders.autoencoder_dc import Decoder
    t = teacher_decoder
    blocks = [type(b[-1]).__name__ for b in t.up_blocks]
    layers = [sum(1 for m in b if type(m).__name__ in ("ResBlock", "EfficientViTBlock")) for b in t.up_blocks]
    s = Decoder(in_channels=3, latent_channels=t.conv_in.in_channels, attention_head_dim=32, block_type=tuple(blocks),
                block_out_channels=tuple(channels), layers_per_block=tuple(layers),
                qkv_multiscales=((), (), (), (5,), (5,), (5,)), norm_type="rms_norm", act_fn="silu",
                upsample_block_type="interpolate", in_shortcut=t.in_shortcut, conv_act_fn="relu")
    s.conv_in.load_state_dict(t.conv_in.state_dict())
    for i in range(len(s.up_blocks)):
        if i not in TRAINED:
            s.up_blocks[i].load_state_dict(t.up_blocks[i].state_dict())
    return s


def load_student_weights(student, ck):
    """A train.py checkpoint ("student": everything) or snapshot ("trained": the student's own stages only)."""
    if "student" in ck:
        student.load_state_dict(ck["student"])
        return
    miss, extra = student.load_state_dict(ck["trained"], strict=False)
    own = tuple(f"up_blocks.{i}." for i in TRAINED) + ("norm_out.", "conv_out.")
    assert not extra and not any(k.startswith(own) for k in miss), (extra, miss[:5])


def prefix(dec, z):
    """Decoder.forward up to the output of up_blocks[3] (512 ch at 1/8 resolution): shared by teacher and student."""
    if dec.in_shortcut:
        h = dec.conv_in(z) + z.repeat_interleave(dec.in_shortcut_repeats, dim=1,
                                                   output_size=z.shape[1] * dec.in_shortcut_repeats)
    else:
        h = dec.conv_in(z)
    for i in reversed(range(len(dec.up_blocks))):
        if i in TRAINED:
            break
        h = dec.up_blocks[i](h)
    return h


def suffix(dec, h):
    for i in reversed(TRAINED):
        h = dec.up_blocks[i](h)
    n = dec.norm_out
    h = h * rms_scale(h, n, inplace=False)
    h = h * n.weight.view(1, -1, 1, 1) + (n.bias.view(1, -1, 1, 1) if n.bias is not None else 0)
    return dec.conv_out(dec.conv_act(h))


def rms_scale(h, norm, inplace=True):
    ss = torch.linalg.vector_norm(h, ord=2, dim=1, keepdim=True, dtype=torch.float32)
    ss = ss.square_() if inplace else ss.square()
    return torch.rsqrt(ss / h.shape[1] + norm.eps).to(h.dtype)


def res_forward_train(self, x):
    """ResBlock with an NCHW RMSNorm, out of place (autograd-safe)."""
    h = self.conv2(F.silu(self.conv1(x)))
    h = h * rms_scale(h, self.norm, inplace=False)
    out = x + h * self.norm.weight.view(1, -1, 1, 1)
    return out + self.norm.bias.view(1, -1, 1, 1) if self.norm.bias is not None else out


def res_forward_fast(self, x):
    """Inference ResBlock: same math, in place (results/2026-10-11_ditic_speed.md)."""
    h = self.conv2(F.silu(self.conv1(x), inplace=True))
    h.mul_(rms_scale(h, self.norm))
    out = torch.addcmul(x, h, self.norm.weight.view(1, -1, 1, 1))
    return out.add_(self.norm.bias.view(1, -1, 1, 1)) if self.norm.bias is not None else out


def patch_resblocks(dec, forward):
    from diffusers.models.autoencoders.autoencoder_dc import ResBlock
    for m in dec.modules():
        if isinstance(m, ResBlock):
            m.forward = types.MethodType(forward, m)

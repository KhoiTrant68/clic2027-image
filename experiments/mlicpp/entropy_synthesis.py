"""Single-pass synthesis conditioned on Gaussian quantization-cell statistics."""
from contextlib import contextmanager
import math

import torch
from torch import nn


def cell_moments(symbol, scale):
    """Approximate mean and variance of the fractional offset, in symbol units."""
    node = math.sqrt(3 / 5) / 2
    scale = scale.clamp_min(0.11)
    precision = scale.reciprocal().square()
    tilt = symbol * (node * precision)
    intercept = -(node * node / 2) * precision + math.log(5 / 8)
    weights = torch.stack((intercept + tilt, torch.zeros_like(tilt), intercept - tilt))
    left, _, right = weights.softmax(dim=0).unbind()
    mean = node * (right - left)
    variance = (node * node * (left + right) - mean.square()).clamp_min(0)
    return mean, variance


class EntropySynthesis(nn.Module):
    def __init__(self, synthesis, channels, mode):
        super().__init__()
        if mode not in {'entropy', 'entropy-blind'}:
            raise ValueError(f'Unsupported entropy synthesis mode: {mode}')
        self.synthesis = synthesis
        self.mode = mode
        self.correct = nn.Sequential(
            nn.Conv2d(channels + 3, 16, 1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16),
            nn.GELU(),
            nn.Conv2d(16, channels, 1),
        )
        nn.init.zeros_(self.correct[-1].weight)
        nn.init.zeros_(self.correct[-1].bias)

    def forward(self, latent, evidence, factor):
        if self.mode == 'entropy-blind':
            evidence = torch.zeros_like(evidence)
        shift = self.correct(torch.cat((latent, evidence), dim=1)).tanh() * (factor / 2)
        return self.synthesis(latent + shift)


@contextmanager
def observe_cells(model, factor):
    """Read pre-LRP dequantized values and scales without altering the context."""
    import models.mlicpp as upstream

    names = ('decompress_anchor', 'decompress_nonanchor')
    original = {name: getattr(upstream, name) for name in names}
    pairs = []
    anchor = None

    def wrap(name):
        def call(gaussian, scales, means, *args):
            nonlocal anchor
            decoded = original[name](gaussian, scales, means, *args)
            if name == names[0]:
                anchor = (decoded - means, scales)
            else:
                if anchor is None:
                    raise RuntimeError('Expected an anchor before the non-anchor')
                residual, anchor_scales = anchor
                symbol = ((residual + decoded - means) / factor).round()
                scale = (anchor_scales + scales) / factor
                mean, variance = cell_moments(symbol, scale)
                pairs.append(torch.stack((mean.sum(1), mean.abs().sum(1), variance.sum(1)), dim=1))
                anchor = None
            return decoded
        return call

    def evidence():
        if len(pairs) != model.slice_num or anchor is not None:
            raise RuntimeError('Incomplete entropy observations')
        return torch.stack(pairs).sum(0) / (model.slice_num * model.slice_ch)

    for name in names:
        setattr(upstream, name, wrap(name))
    try:
        yield evidence
    finally:
        for name, function in original.items():
            setattr(upstream, name, function)


def decode(model, strings, shape, factor=1.0):
    synthesis = model.g_s
    if not isinstance(synthesis, EntropySynthesis):
        return model.decompress(strings, shape)
    if synthesis.mode == 'entropy-blind':
        hook = synthesis.register_forward_pre_hook(
            lambda _, args: (args[0], torch.zeros_like(args[0][:, :3]), factor))
        try:
            return model.decompress(strings, shape)
        finally:
            hook.remove()
    with observe_cells(model, factor) as evidence:
        hook = synthesis.register_forward_pre_hook(lambda _, args: (args[0], evidence(), factor))
        try:
            return model.decompress(strings, shape)
        finally:
            hook.remove()

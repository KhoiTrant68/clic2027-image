"""Experimental diagonal quantization, using the pinned MLIC checkerboard codec.

Only quantization/dequantization and Gaussian scale units change; upstream is
never edited. The context hook resets for each encode/decode from decoded z.
"""
import struct
import numpy as np
import torch
from torch import nn
from stream import pack, unpack

MODES = ['uniform', 'global', 'predicted', 'oracle']
HEADER = struct.Struct('>4sBBfH')


def centered(v):
    return v - v.mean(-1, keepdim=True)


def relative(log_metric):
    v = -.5 * centered(log_metric)
    lo, hi = v.min(-1, keepdim=True).values-1, v.max(-1, keepdim=True).values+1
    for _ in range(32):
        mid = (lo+hi)/2
        positive = (v-mid).clamp(-np.log(2), np.log(2)).mean(-1, keepdim=True) > 0
        lo, hi = torch.where(positive, mid, lo), torch.where(positive, hi, mid)
    return (v-(lo+hi)/2).clamp(-np.log(2), np.log(2)).exp()


def features(h):
    return torch.cat([h.mean((2, 3)), h.std((2, 3), unbiased=False)], 1)


class Predictor(nn.Module):
    def __init__(self, mean, std, target):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.register_buffer('target', target)
        self.net = nn.Sequential(nn.Linear(1280, 128), nn.GELU(), nn.Linear(128, 320))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        f = features(h) if h.ndim == 4 else h
        return centered(self.target + self.net((f-self.mean)/self.std))


class Quantizer:
    def __init__(self, model, predictor, global_metric):
        import models.mlicpp as module
        self.module, self.model = module, model
        self.predictor, self.global_metric = predictor, global_metric
        self.mode, self.factor, self.oracle = 'uniform', 1., None
        self.original = {n: getattr(module, n) for n in
                         ['compress_anchor', 'compress_nonanchor', 'decompress_anchor', 'decompress_nonanchor']}
        for name in self.original:
            setattr(module, name, self.wrapper(name))
        self.hook = model.h_s.register_forward_hook(self.context)

    def context(self, module, args, h):
        self.hyper = h.detach()
        self.cursor = 0
        if self.mode == 'uniform':
            self.delta = torch.ones((1, 320), device=h.device) * self.factor
        elif self.mode == 'global':
            self.delta = relative(self.global_metric) * self.factor
        elif self.mode == 'predicted':
            self.delta = relative(self.predictor(h)) * self.factor
        else:
            self.delta = self.oracle.to(h.device) * self.factor
        assert torch.isfinite(self.delta).all() and self.delta.min() > 0

    def wrapper(self, name):
        original = self.original[name]
        def call(gc, *args):
            i = self.cursor // 2
            self.cursor += 1
            d = self.delta[:, i*32:(i+1)*32, None, None]
            # Pinned checkpoint uses ten 32-channel slices.
            assert d.shape[1] == self.model.slice_ch == 32
            if name.startswith('compress'):
                y, s, m, symbols, indexes = args
                if not torch.isfinite(y).all() or ((y-m)/d).abs().max() >= 2**24:
                    raise ValueError('Unsafe native entropy symbol')
                return original(gc, y/d, s/d, m/d, symbols, indexes)*d
            s, m, decoder, cdf, lengths, offsets = args
            return original(gc, s/d, m/d, decoder, cdf, lengths, offsets)*d
        return call

    def select(self, mode, factor, oracle=None):
        self.mode, self.factor, self.oracle = mode, float(np.float32(factor)), oracle

    def close(self):
        self.hook.remove()
        for name, original in self.original.items():
            setattr(self.module, name, original)


def envelope(encoded, mode, factor, oracle=None):
    side = b'' if oracle is None else np.asarray(oracle, dtype='<f4').reshape(320).tobytes()
    return HEADER.pack(b'GEO1', 1, MODES.index(mode), factor, len(side)) + side + pack(
        2, True, (256, 256), tuple(encoded['shape']), encoded['strings'])


def read(blob):
    magic, version, mode, factor, n = HEADER.unpack_from(blob)
    assert magic == b'GEO1' and version == 1 and mode < 4 and factor > 0
    assert n == (1280 if MODES[mode] == 'oracle' else 0)
    side = torch.from_numpy(np.frombuffer(blob[HEADER.size:HEADER.size+n], dtype='<f4').copy())[None] if n else None
    q, legacy, size, shape, strings = unpack(blob[HEADER.size+n:])
    assert q == 2 and legacy and size == (256, 256)
    return MODES[mode], factor, side, shape, strings

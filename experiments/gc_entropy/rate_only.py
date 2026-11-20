"""Phase 0 trainer (plans/GC_entropy_MLICpp_design_2026-11-20.md): entropy-model variants on frozen MLIC++ latents.

    python rate_only.py --variant V0|V0f|V1|V2|V3|V3s --data DIR(extract.py output) --out DIR [--steps 50000 --batch 32]

y = g_a(x) of a frozen MLIC++ checkpoint is fixed, so the loss is the rate alone (bits of y under additive noise plus
bits of z) and every bpp difference between variants comes from the entropy model. Each variant has the GC
structure (src/clic27/gc/model.py): GC hyper-analysis h_a, zero-mean Gaussian z prior, integer h_s, uneven channel
slices with a two-pass checkerboard, and integer (QConv2d, straight-through quantisers) parameter nets ending in
ScaleMeanHead (dyadic mean, 4 fractional bits). Additions, each behind a flag:
  local5  E1: full 5x5 integer conv on the decoded anchors of the slice (instead of raw anchors into depthwise 3x3)
  deep    E2: deeper 1x1 parameter net (256 -> 192 -> 128), one depthwise 3x3
  inter   E3: integer linear attention over all positions of the previous slices (anchor and non-anchor)
  intra   E4: integer linear attention, map learnt on the previous slice (query: all positions, key: its anchors)
              applied to the anchors of the current slice (non-anchor pass)
  integer False: same nets in float (no weight/activation/mean quantisation): the cost of integerisation.
The attention follows the integer formula of the design (ReLU features with 4 fractional bits capped at 8 bits,
signed values capped at 10 bits, out = floor(16 * sum_i q_i KV_ij / (sum_i q_i Ks_i + 1/256)) / 16) with
straight-through floors; the bit-exact forward_int is phase 1. Scores: estimated bits (discrete Gaussian of
round(y - mu) + mu, discrete z) of the 30 CLIC images, i.e. the same quantity as extract.py's MLIC++ reference.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments/new_codecs'))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from clic27.entropy.intnet import IntSequential, QConv2d, Up2, ste  # noqa: E402
from clic27.entropy.models import ScaleMeanHead, gaussian_bits  # noqa: E402
from clic27.entropy import tables  # noqa: E402
from clic27.gc.layers import HyperAnalysis  # noqa: E402
from clic27.gc.model import FRAC, Y_CLAMP, Z_CLAMP, ZPrior, checkerboard  # noqa: E402

GC_SLICES = [16, 16, 32, 64, 192]
BASE = dict(integer=True, local5=False, deep=False, inter=False, intra=False, slices=GC_SLICES)
VARIANTS = {
    'V0': BASE,
    'V0f': {**BASE, 'integer': False},
    'V1': {**BASE, 'local5': True, 'deep': True},
    'V2': {**BASE, 'local5': True, 'deep': True, 'inter': True},
    'V3': {**BASE, 'local5': True, 'deep': True, 'inter': True, 'intra': True},
    'V3s': {**BASE, 'local5': True, 'deep': True, 'inter': True, 'intra': True, 'slices': [32] * 10},
}
G = 64  # attention width (2 heads x 32) and output channels of each global context


class FConv2d(QConv2d):
    """QConv2d without any quantisation (float variant)."""

    def forward(self, x, cond=None):
        y = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, 1, self.groups)
        return y if self.act is None else F.relu(y)


class FHead(nn.Module):
    def forward(self, acc):
        a_s, a_m = acc.chunk(2, dim=1)
        return float(tables.scale_table()[0]) + F.softplus(a_s), a_m


def param_net(Conv, cin, cout, deep):
    q = dict(in_frac=FRAC, out_frac=FRAC)
    if deep:
        return IntSequential(Conv(cin, 256, 1, **q), Conv(256, 256, 3, groups=256, **q), Conv(256, 192, 1, **q),
                             Conv(192, 128, 1, **q), Conv(128, 2 * cout, 1, in_frac=FRAC, act=None))
    h = 160  # = GC ctx_net (model.ctx_hidden)
    return IntSequential(Conv(cin, h, 1, **q), Conv(h, h, 3, groups=h, **q), Conv(h, h, 3, groups=h, **q),
                         Conv(h, h, 1, **q), Conv(h, 2 * cout, 1, in_frac=FRAC, act=None))


class LinAttn(nn.Module):
    """Linear attention as the integer design: q, k >= 0 (8-bit, 4 frac bits), v signed (10-bit, 4 frac bits)."""

    def __init__(self, cq, cv, integer, d=G, heads=2, cout=G):
        super().__init__()
        Conv = QConv2d if integer else FConv2d
        q = dict(in_frac=FRAC, out_frac=FRAC)
        self.integer, self.h, self.d = integer, heads, d
        self.q = IntSequential(Conv(cq, d, 1, **q), Conv(d, d, 3, groups=d, a_bits=8, **q))
        self.k = IntSequential(Conv(cq, d, 1, **q), Conv(d, d, 3, groups=d, a_bits=8, **q))
        self.v = Conv(cv, d, 3, in_frac=FRAC, act=None)
        self.proj = Conv(d, cout, 5, **q)

    def forward(self, xq, xv, mk):
        q, k, v = self.q(xq), self.k(xq) * mk, self.v(xv)
        if self.integer:
            v = ste(torch.clamp(torch.floor(v * 16), -512, 511) / 16, v)
        v = v * mk
        B, _, H, W = q.shape
        q, k, v = (t.reshape(B, self.h, self.d // self.h, H * W) for t in (q, k, v))
        kv = torch.einsum('bhin,bhjn->bhij', k, v)
        num = torch.einsum('bhin,bhij->bhjn', q, kv)
        den = torch.einsum('bhin,bhi->bhn', q, k.sum(-1)).unsqueeze(2) + 1 / 256
        o = num / den
        if self.integer:
            o = ste(torch.floor(o * 16) / 16, o)
        return self.proj(o.reshape(B, self.d, H, W))


class RateModel(nn.Module):
    def __init__(self, v, M=320, Z=192, H=192):
        super().__init__()
        self.v, self.slices = dict(v), list(v['slices'])
        assert sum(self.slices) == M
        Conv = QConv2d if v['integer'] else FConv2d
        self.h_a = HyperAnalysis(M, Z)
        self.prior = ZPrior(Z)
        self.h_s = IntSequential(
            Conv(Z, H, 3, in_frac=0, out_frac=FRAC), Up2(),
            Conv(H, H, 3, in_frac=FRAC, out_frac=FRAC), Up2(),
            Conv(H, H, 3, in_frac=FRAC, out_frac=FRAC))
        self.head = ScaleMeanHead(acc_frac=7 + FRAC, mu_frac=FRAC) if v['integer'] else FHead()
        self.pa, self.pn, self.loc, self.inter, self.intra = (nn.ModuleList() for _ in range(5))
        prev = 0
        for k, c in enumerate(self.slices):
            gi = G if v['inter'] and k else 0
            gr = G if v['intra'] and k else 0
            L = 2 * c if v['local5'] else c
            self.pa.append(param_net(Conv, H + prev + gi, c, v['deep']))
            self.pn.append(param_net(Conv, H + prev + L + gi + gr, c, v['deep']))
            self.loc.append(Conv(c, 2 * c, 5, in_frac=FRAC, out_frac=FRAC) if v['local5'] else nn.Identity())
            self.inter.append(LinAttn(prev, prev, v['integer']) if gi else nn.Identity())
            self.intra.append(LinAttn(self.slices[k - 1], c, v['integer']) if gr else nn.Identity())
            prev += c

    def forward(self, y, noise=True):
        """y (B, M, h, w), h and w multiples of 4. Returns total bits per item (B,)."""
        z = self.h_a(y)
        z_hat, bz = self.prior(z, noise)
        hyp = self.h_s(torch.clamp(z_hat, -Z_CLAMP, Z_CLAMP))
        B, _, h, w = y.shape
        ma = checkerboard(h, w, y.device)
        mn, ones = 1 - ma, torch.ones_like(ma)
        prev, total, start = [], bz.flatten(1).sum(1), 0
        for k, c in enumerate(self.slices):
            yk = y[:, start:start + c]
            start += c
            P = [torch.cat(prev, 1)] if prev else []
            g = [self.inter[k](P[0], P[0], ones)] if self.v['inter'] and k else []
            sig_a, mu_a = self.head(self.pa[k](torch.cat([hyp, *P, *g], 1)))
            ya = ste(torch.round(yk - mu_a) + mu_a, yk)
            ctx = torch.clamp(ya * ma, -Y_CLAMP, Y_CLAMP)
            loc = self.loc[k](ctx)
            r = [self.intra[k](prev[-1], ctx, ma)] if self.v['intra'] and k else []
            sig_n, mu_n = self.head(self.pn[k](torch.cat([hyp, *P, loc, *g, *r], 1)))
            yn = ste(torch.round(yk - mu_n) + mu_n, yk)
            if noise:
                yt = yk + torch.empty_like(yk).uniform_(-0.5, 0.5)
                b = gaussian_bits(yt, sig_a, mu_a) * ma + gaussian_bits(yt, sig_n, mu_n) * mn
            else:
                b = gaussian_bits(ya, sig_a, mu_a) * ma + gaussian_bits(yn, sig_n, mu_n) * mn
            total = total + b.flatten(1).sum(1)
            prev.append(torch.clamp(ya * ma + yn * mn, -Y_CLAMP, Y_CLAMP))
        return total


@torch.no_grad()
def evaluate(model, clic):
    model.eval()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    per = []
    for y, r in zip(clic['y'], clic['rows']):
        bits = float(model(y.cuda(), noise=False)[0])
        per.append(dict(name=r['name'], bits=bits, bpp=bits / r['pixels'], mlicpp_bpp=r['mlicpp_est_bits'] / r['pixels']))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    model.train()
    px = sum(r['pixels'] for r in clic['rows'])
    return dict(bpp=sum(p['bits'] for p in per) / px, mlicpp_bpp=sum(r['mlicpp_est_bits'] for r in clic['rows']) / px,
                forward_s=dt, images=per)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=sorted(VARIANTS), required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--eval-every', type=int, default=20000)
    p.add_argument('--ckpt-every', type=int, default=5000)
    p.add_argument('--init', type=Path, help='start from the weights of another run (model.pt or last.pt); fresh '
                                             'optimiser and step counter (design 2.1: long runs from the 50k models)')
    a = p.parse_args()
    from half_vram import install
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / 'result.json').exists():
        print('already done', flush=True)
        return
    stop = install(a.out / 'vram.jsonl')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(1)
    pool = torch.load(a.data / 'pool.pt')['y']
    clic = torch.load(a.data / 'clic30.pt')
    model = RateModel(VARIANTS[a.variant]).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    step, last, init_step = 0, a.out / 'last.pt', 0
    if a.init:  # weights only when starting; the init step is recorded on resume too
        init_step = int(json.loads((a.init.parent / 'result.json').read_text())['steps'])
        if not last.exists():
            s = torch.load(a.init, map_location='cpu')
            if s.get('variant', a.variant) != a.variant:
                raise SystemExit(f"--init is variant {s['variant']}, not {a.variant}")
            model.load_state_dict(s['model'])
            print(f'init from {a.init} (trained {init_step} steps)', flush=True)
    if last.exists():
        s = torch.load(last, map_location='cpu')
        model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); step = s['step']
        print(f'resume from step {step}', flush=True)
    n_par = sum(p.numel() for p in model.parameters())
    (a.out / 'meta.json').write_text(json.dumps(dict(variant=a.variant, config=VARIANTS[a.variant], params=n_par,
                                                     steps=a.steps, batch=a.batch, lr=a.lr, pool=tuple(pool.shape),
                                                     data=str(a.data), init=str(a.init) if a.init else None,
                                                     init_step=init_step), indent=1))
    gen = torch.Generator().manual_seed(1000 + step)
    px = pool.shape[-1] * pool.shape[-2] * 256  # pixels of one crop (latent at 1/16)
    logf = (a.out / 'log.jsonl').open('a')
    tl, acc = time.time(), []
    while step < a.steps:
        for g in opt.param_groups:
            g['lr'] = a.lr * (0.1 if step >= 0.9 * a.steps else 1.0)
        y = pool[torch.randint(len(pool), (a.batch,), generator=gen)].cuda(non_blocking=True).float()
        loss = model(y).mean() / px
        if not torch.isfinite(loss):
            raise SystemExit(f'non-finite loss at step {step}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        acc.append(float(loss))
        if step % 500 == 0 or step == 1:
            rec = dict(step=step, bpp=sum(acc) / len(acc), gnorm=float(gn), it_s=len(acc) / (time.time() - tl))
            acc, tl = [], time.time()
            if step % a.eval_every == 0:
                e = evaluate(model, clic)
                rec.update(clic_bpp=e['bpp'], mlicpp_bpp=e['mlicpp_bpp'])
                tl = time.time()
            print(json.dumps(rec), flush=True)
            logf.write(json.dumps(rec) + '\n'); logf.flush()
        if step % a.ckpt_every == 0 or step == a.steps:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step), last.with_suffix('.tmp'))
            last.with_suffix('.tmp').replace(last)
    e = evaluate(model, clic)
    evaluate(model, clic)  # second pass: the timing of the first includes kernel warm-up
    e['forward_s'] = evaluate(model, clic)['forward_s']
    e.update(variant=a.variant, config=VARIANTS[a.variant], params=n_par, steps=init_step + step, init_step=init_step,
             gain_vs_mlicpp_pct=100 * (e['bpp'] / e['mlicpp_bpp'] - 1))
    torch.save(dict(model=model.state_dict(), variant=a.variant), a.out / 'model.pt')
    (a.out / 'result.json').write_text(json.dumps(e, indent=1))
    print(f"RESULT {a.variant} clic30 {e['bpp']:.4f} bpp vs MLIC++ {e['mlicpp_bpp']:.4f} "
          f"({e['gain_vs_mlicpp_pct']:+.2f}%) forward {e['forward_s']:.2f} s", flush=True)
    stop.set()


if __name__ == '__main__':
    main()

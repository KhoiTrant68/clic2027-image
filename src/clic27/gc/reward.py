"""Q-hat v2: a differentiable human-preference distance fitted on CLIC pairwise ratings.

d(O, X) = pool_tau( sum_l w_l . (phi_l(O) - phi_l(X))^2  +  w_wd * WD-map(O, X) )
  phi_l: channel-normalised VGG16 features at 5 depths (as LPIPS), w_l >= 0 learned 1x1 weights,
  WD-map: local mean/std feature differences (Wasserstein-distortion style, 2 window sizes),
  pool_tau: (1/tau) log mean exp(tau * map) over space; tau = 0 is the mean, tau -> inf the worst region.
P(A preferred over B | O) = sigmoid(alpha * (d(O, B) - d(O, A)))     (Bradley-Terry, no bias: order-symmetric)

Only w, w_wd, tau and alpha are trained (VGG frozen), so a few thousand CLIC questions suffice and the
distance stays usable as a training loss and as the crop scorer of the allocator.

    python -m clic27.gc.reward fetch --out DIR [--n 4000]   # crops of sampled questions -> DIR/crops, index.jsonl
    python -m clic27.gc.reward fit   --out DIR               # fit, held-out accuracy -> DIR/reward.pt, report.json
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO = Path(__file__).resolve().parents[3]


class PrefDistance(nn.Module):
    LAYERS = (3, 8, 15, 22, 29)  # relu1_2 ... relu5_3
    CH = (64, 128, 256, 512, 512)

    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision
        from torchvision.models.vgg import VGG16_Weights
        w = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vgg = torchvision.models.vgg16(weights=w).features[: max(self.LAYERS) + 1].eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.w_raw = nn.ParameterList([nn.Parameter(torch.full((c,), -3.0)) for c in self.CH])  # softplus -> ~0.05
        self.wd_raw = nn.Parameter(torch.tensor([-3.0, -3.0]))
        self.tau_raw = nn.Parameter(torch.tensor(-2.0))     # softplus -> tau ~ 0.13
        self.alpha_raw = nn.Parameter(torch.tensor(2.0))
        k = torch.tensor([1.0, 4, 6, 4, 1])
        self.register_buffer("blur", (torch.outer(k, k) / 256).view(1, 1, 5, 5))

    def train(self, mode=True):
        super().train(mode)
        self.vgg.eval()
        return self

    @property
    def tau(self):
        return F.softplus(self.tau_raw)

    @property
    def alpha(self):
        return F.softplus(self.alpha_raw)

    def _feats(self, x):
        x = (x - self.mean) / self.std
        out = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.LAYERS:
                out.append(x)
        return out

    def _local_stats(self, f, it):
        B, C, H, W = f.shape
        m, q = f.reshape(B * C, 1, H, W), (f * f).reshape(B * C, 1, H, W)
        for _ in range(it):
            m, q = F.conv2d(m, self.blur, padding=2), F.conv2d(q, self.blur, padding=2)
        return m.reshape(B, C, H, W), (q - m * m).reshape(B, C, H, W).clamp_min(1e-6).sqrt()

    def dist_map(self, o, x):
        """o, x: (B, 3, H, W) in [0, 1]. Returns (B, 1, H/2, W/2) distance map."""
        fo, fx = self._feats(o), self._feats(x)
        size = (o.shape[-2] // 2, o.shape[-1] // 2)
        tot = 0.0
        for l, (a, b) in enumerate(zip(fo, fx)):
            na = a / (a.pow(2).sum(1, keepdim=True).sqrt() + 1e-10)
            nb = b / (b.pow(2).sum(1, keepdim=True).sqrt() + 1e-10)
            d = ((na - nb) ** 2 * F.softplus(self.w_raw[l]).view(1, -1, 1, 1)).sum(1, keepdim=True)
            if l in (1, 2):  # texture statistics at two window sizes on mid layers
                for j, it in enumerate((2, 6)):
                    ma, sa = self._local_stats(a, it)
                    mb, sb = self._local_stats(b, it)
                    s = ((ma - mb) ** 2 + (sa - sb) ** 2).mean(1, keepdim=True) / (a.pow(2).mean() + 1e-6)
                    d = d + F.softplus(self.wd_raw[j]) * s
            tot = tot + F.interpolate(d, size=size, mode="bilinear", align_corners=False)
        return tot

    def pool(self, m):
        t = self.tau
        flat = m.flatten(1)
        return (torch.logsumexp(t * flat, 1) - math.log(flat.shape[1])) / t.clamp_min(1e-4)

    def forward(self, o, x):
        """Scalar distance per sample (B,); lower = closer to what raters prefer."""
        return self.pool(self.dist_map(o, x))

    def logit_a(self, o, a, b):
        return self.alpha * (self.forward(o, b) - self.forward(o, a))


def load_reward(path, dev="cpu") -> PrefDistance:
    """Load a fitted Q-hat v2. reward.pt carries the VGG features too, so loading needs no download and
    scores exactly as at fit time. Older files without VGG weights get the ImageNet VGG (the one fit used)."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd["state_dict"] if "state_dict" in sd else sd
    has_vgg = any(k.startswith("vgg.") for k in sd)
    if not has_vgg:
        print(f"[reward] {path} has no VGG weights (old format): using torchvision ImageNet VGG16", file=sys.stderr)
    m = PrefDistance(pretrained=not has_vgg)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    bad = [k for k in missing if has_vgg or not k.startswith("vgg.")] + list(unexpected)
    if bad:
        raise RuntimeError(f"reward checkpoint {path} does not match PrefDistance: {bad[:8]}")
    return m.to(dev).eval()


# ---------------------------------------------------------------- data: CLIC rated crops
def _qhat_module():
    sys.path.insert(0, str(REPO / "experiments" / "qhat"))
    sys.path.insert(0, str(REPO / "src"))
    import qhat  # noqa: E402
    return qhat


def fetch(a):
    """Download (range requests) the crops of n sampled questions per set into out/crops; write index.jsonl."""
    from PIL import Image
    q = _qhat_module()
    out = Path(a.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    idx_path = out / "index.jsonl"
    done = set()
    if idx_path.exists():
        done = {(r["set"], r["row"]) for r in map(json.loads, idx_path.read_text().splitlines())}
    sets = a.sets or list(q.SETS)
    with open(idx_path, "a") as fh:
        for name in sets:
            rows = q.sample(q.load_answers(name, out / "ratings"), a.n, a.seed)
            crops = q.Crops(q.SETS[name][2])
            todo = [r for r in rows if (name, r["row"]) not in done]
            print(f"[fetch] {name}: {len(todo)} questions to fetch", flush=True)
            from concurrent.futures import ThreadPoolExecutor
            cache = {}

            def get(key):
                if key not in cache:
                    p = out / "crops" / (key.replace("/", "__").rsplit(".", 1)[0] + ".png")
                    if not p.exists():
                        Image.fromarray(crops.fetch(key)).save(p)
                    cache[key] = p.name
                return cache[key]

            with ThreadPoolExecutor(a.threads) as pool:
                for r, names in zip(todo, pool.map(lambda r: [get(r["o"]), get(r["a"]), get(r["b"])], todo)):
                    fh.write(json.dumps(dict(set=name, row=r["row"], o=names[0], a=names[1], b=names[2],
                                             ans=r["ans"])) + "\n")
                    fh.flush()


class PairSet(torch.utils.data.Dataset):
    def __init__(self, root, rows, crop):
        self.root, self.rows, self.crop = Path(root) / "crops", rows, crop

    def __len__(self):
        return len(self.rows)

    def _img(self, name):
        from PIL import Image
        return torch.from_numpy(np.array(Image.open(self.root / name).convert("RGB"))).permute(2, 0, 1).float() / 255

    def __getitem__(self, i):
        r = self.rows[i]
        o, a, b = self._img(r["o"]), self._img(r["a"]), self._img(r["b"])
        H = min(t.shape[1] for t in (o, a, b))
        W = min(t.shape[2] for t in (o, a, b))
        c = min(self.crop, H, W) // 16 * 16
        y, x = random.randint(0, H - c), random.randint(0, W - c)
        sl = (slice(None), slice(y, y + c), slice(x, x + c))
        return o[sl], a[sl], b[sl], torch.tensor(float(r["ans"] == 0))


def fit(a):
    out = Path(a.out)
    rows = [json.loads(l) for l in (out / "index.jsonl").read_text().splitlines()]
    test_sets = set(a.test_sets)
    tr = [r for r in rows if r["set"] not in test_sets]
    te = [r for r in rows if r["set"] in test_sets]
    if not te:  # smoke runs may fetch a single set: hold out 20%
        random.Random(0).shuffle(rows)
        k = max(1, len(rows) // 5)
        te, tr = rows[:k], rows[k:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = PrefDistance().to(dev)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr)
    dl = torch.utils.data.DataLoader(PairSet(out, tr, a.crop), batch_size=a.batch, shuffle=True, num_workers=4,
                                     drop_last=len(tr) > a.batch)

    def evaluate(rs):
        m.eval()
        ok = n = 0
        ll = 0.0
        with torch.no_grad():
            for o, x, y, t in torch.utils.data.DataLoader(PairSet(out, rs, 10 ** 9), batch_size=1):
                lg = m.logit_a(o.to(dev), x.to(dev), y.to(dev)).cpu()
                ok += int((lg > 0).float().eq(t).sum())
                ll += float(F.binary_cross_entropy_with_logits(lg, t, reduction="sum"))
                n += 1
        m.train()
        return dict(acc=ok / max(n, 1), nll=ll / max(n, 1), n=n)

    init = evaluate(te)
    print(f"[reward] init (uniform weights) test {init}", flush=True)
    step = 0
    while step < a.steps:
        for o, x, y, t in dl:
            lg = m.logit_a(o.to(dev), x.to(dev), y.to(dev))
            loss = F.binary_cross_entropy_with_logits(lg, t.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 100 == 0:
                print(f"[reward] step {step} loss {loss.item():.4f} tau {m.tau.item():.3f}", flush=True)
            if step >= a.steps:
                break
    res = evaluate(te)
    report = dict(train=len(tr), test=len(te), test_sets=sorted(test_sets), init=init, final=res,
                  tau=m.tau.item(), alpha=m.alpha.item(), note="per-judgment accuracy on held-out questions")
    # Keep the VGG features in the file (~60 MB): load_reward must reproduce fit-time scores offline.
    torch.save(dict(state_dict=m.state_dict(), report=report), out / "reward.pt")
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--out", required=True)
    f.add_argument("--sets", nargs="*")
    f.add_argument("--n", type=int, default=4000)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--threads", type=int, default=16)
    g = sub.add_parser("fit")
    g.add_argument("--out", required=True)
    g.add_argument("--test-sets", nargs="*", default=["2024t"])
    g.add_argument("--steps", type=int, default=4000)
    g.add_argument("--batch", type=int, default=16)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--crop", type=int, default=256)
    a = ap.parse_args(argv)
    {"fetch": fetch, "fit": fit}[a.cmd](a)


if __name__ == "__main__":
    main()

"""Q-hat: a stand-in for CLIC's human raters, fitted on the public CLIC perceptual ratings.

v0 (2026-10-06): PSNR, MS-SSIM, LPIPS alex/vgg, DISTS, CLIP-IQA. v1 adds Wasserstein Distortion at two pooling
scales (wd3, wd5: log2 sigma 3 and 5), see wasserstein.py.

Each rating row is (O, A, B, answer): a rater saw the original O and two reconstructions, answer 0 = A
preferred, 1 = B. With metric features phi(O, X), a Bradley-Terry / logistic model on feature differences

    P(A preferred) = sigmoid(b + w . (phi(O, A) - phi(O, B)))

gives a per-image utility Q-hat(O, X) = w . phi(O, X) that we can compute for our own reconstructions.

    python qhat.py features --out work/qhat [--n 4000 ...]   # GPU: fetch sampled crops, compute features
    python qhat.py fit --out work/qhat                        # CPU/numpy: fit, held-out accuracy, summary.md
    python qhat.py add --out work/qhat                        # GPU: features missing from features.csv (e.g.
                                                              # wd3/wd5 for a v0 run), for the same questions

Crops are read straight out of the remote zips with HTTP range requests (remote_zip.py), so nothing like
the 46-135 GiB archives is downloaded. Features are appended to features.csv, so re-runs resume.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sys
import threading
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

BASE = "https://downloads.compression.cc/"
# name: (ratings zip, answers csv inside it, crops zip, default sample size)
SETS = {
    "2021v": ("clic2021_perceptual_valid_ratings", "clic2021_perceptual_valid_answers.csv",
              "clic2021_perceptual_valid_crops", 5220),
    "2021t": ("clic2021_perceptual_test_ratings", "clic2021_perceptual_test_answers.csv",
              "clic2021_perceptual_test_crops", 3000),
    "2022t": ("clic2022_perceptual_test_ratings", "clic2022_perceptual_test_answers.csv",
              "clic2022_perceptual_test_crops", 3000),
    "2024t": ("clic2024_perceptual_image_test_ratings", "clic2024_perceptual_image_test_answers.csv",
              "clic2024_perceptual_image_test_crops", 5000),
}
FULL_REF = ["psnr", "msssim", "lpips_alex", "lpips_vgg", "dists", "wd3", "wd5"]
WD = {"wd3": 3, "wd5": 5}  # Wasserstein feature -> log2 sigma
NO_REF = ["clipiqa"]  # of the reconstruction only; skipped if the weights cannot be loaded


# ---------------------------------------------------------------- ratings
def load_answers(name, cache: Path):
    rz, csv_name, _, _ = SETS[name]
    p = cache / f"{rz}.zip"
    if not p.exists():
        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BASE + rz + ".zip", p)
    with zipfile.ZipFile(p) as z, z.open(csv_name) as f:
        rows = [r for r in csv.reader(io.TextIOWrapper(f)) if len(r) == 4]
    return [dict(o=o, a=a, b=b, ans=int(ans)) for o, a, b, ans in rows]


def questions(rows):
    """Rows that compare two different reconstructions (drops sanity rows where the original is a choice).
    The `row` ids in features.csv index this list."""
    return [r for r in rows if r["o"] not in (r["a"], r["b"]) and r["a"] != r["b"]]


def sample(rows, n, seed):
    """Seeded sample without replacement over questions(rows)."""
    rows = questions(rows)
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    return [dict(rows[i], row=i) for i in sorted(idx[:n])]


def sanity_stats(rows):
    """Rows where O is one of A/B: how often raters picked the original (a rater-quality check)."""
    s = [r for r in rows if r["o"] in (r["a"], r["b"]) and r["a"] != r["b"]]
    ok = sum((r["ans"] == 0) == (r["a"] == r["o"]) for r in s)
    return dict(n=len(s), picked_original=ok / len(s) if s else float("nan"))


def duplicate_agreement(rows):
    """Questions asked more than once (same O and same unordered pair): fraction of answer pairs that agree.
    Upper bound on the accuracy any predictor can reach on this data."""
    groups = {}
    for r in rows:
        a, b = sorted((r["a"], r["b"]))
        pick = r["a"] if r["ans"] == 0 else r["b"]
        groups.setdefault((r["o"], a, b), []).append(pick)
    agree = total = 0
    for picks in groups.values():
        for i in range(len(picks)):
            for j in range(i + 1, len(picks)):
                agree += picks[i] == picks[j]
                total += 1
    return dict(pairs=total, agreement=agree / total if total else float("nan"))


# ---------------------------------------------------------------- remote crops
class Crops:
    """Member lookup in a remote crop zip + parallel single-request member fetches."""

    def __init__(self, crops_zip):
        from remote_zip import RemoteZip
        self.url = BASE + crops_zip + ".zip"
        with RemoteZip(self.url) as z:
            self.info = {i.filename: i for i in z.infolist() if not i.is_dir()}
        self.by_base = {}
        for k in self.info:
            self.by_base.setdefault(k.rsplit("/", 1)[-1], k)
        self.lock, self.bytes = threading.Lock(), 0

    def key(self, name):
        if name in self.info:
            return name
        k = self.by_base.get(name.rsplit("/", 1)[-1])
        if k is None:
            raise KeyError(f"{name} not in {self.url}")
        return k

    def fetch(self, name) -> np.ndarray:
        from PIL import Image
        i = self.info[self.key(name)]
        lo = i.header_offset
        hi = lo + 30 + len(i.orig_filename.encode()) + 1024 + i.compress_size  # slack for the local extra field
        for k in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    buf = r.read()
                break
            except OSError:
                if k == 4:
                    raise
                time.sleep(2 ** k)
        n, m = int.from_bytes(buf[26:28], "little"), int.from_bytes(buf[28:30], "little")
        data = buf[30 + n + m:30 + n + m + i.compress_size]
        if i.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompressobj(-15).decompress(data)
        elif i.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"compression {i.compress_type} not supported")
        with self.lock:
            self.bytes += len(buf)
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def bounded(pool, fn, items, window):
    """Yield futures of fn(item) in order, with at most `window` submitted but not yet consumed."""
    from collections import deque
    q, it = deque(), iter(items)
    for x in it:
        q.append(pool.submit(fn, x))
        if len(q) >= window:
            yield q.popleft()
    while q:
        yield q.popleft()


# ---------------------------------------------------------------- features
class Features:
    def __init__(self, dev):
        import lpips
        from clic27.eval import metrics as M
        self.M, self.dev = M, dev
        self.per = M.Perceptual(dev)
        self._vgg = lpips.LPIPS(net="vgg", verbose=False).to(dev)
        from wasserstein import WassersteinDistortion
        self._wd = WassersteinDistortion(dev)
        self._clipiqa = None
        try:
            import piq
            self._clipiqa = piq.CLIPIQA(data_range=1.0).to(dev)
        except Exception as e:  # noqa: BLE001  optional feature
            print("CLIP-IQA unavailable, skipping:", e, flush=True)

    def names(self):
        return FULL_REF + (NO_REF if self._clipiqa is not None else [])

    def __call__(self, o, x, only=None) -> dict:
        """Every feature, or only those named in `only`."""
        import torch
        M = self.M
        want = set(only or self.names())
        f = {}
        if want & {"psnr", "msssim", "lpips_alex", "lpips_vgg", "dists"}:
            m = M.mse(o, x)
            f = dict(psnr=M.psnr_from_mse(m), msssim=M.msssim(o, x, self.dev), lpips_alex=self.per.lpips(o, x),
                     lpips_vgg=M.tiled(lambda a, b: self._vgg(a * 2 - 1, b * 2 - 1).mean(), o, x, self.dev),
                     dists=self.per.dists(o, x))
        if want & set(WD):
            wd = self._wd(o, x, tuple(WD.values()))
            f.update({k: wd[s] for k, s in WD.items()})
        f = {k: v for k, v in f.items() if k in want}
        if self._clipiqa is not None and "clipiqa" in want:
            with torch.no_grad():
                f["clipiqa"] = float(self._clipiqa(M.to_tensor(x, self.dev)).mean())
        return f


def features(a):
    from clic27.eval import metrics as M
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = M.device()
    if dev == "cpu" and not a.allow_cpu:
        raise SystemExit("features needs a GPU (or --allow-cpu)")
    feat = Features(dev)
    path = out / "features.csv"
    done = set()
    if path.exists():
        done = {(r["set"], int(r["row"])) for r in M.read_csv(path)}
    cols = ["set", "row", "ans"] + [f"{k}_{s}" for s in "ab" for k in feat.names()]
    new = not path.exists()
    fh = open(path, "a", newline="")
    w = csv.DictWriter(fh, cols)
    if new:
        w.writeheader()
    meta = {}
    for name in a.sets:
        rows = load_answers(name, out / "ratings")
        meta[name] = dict(rows=len(rows), sanity=sanity_stats(rows), duplicates=duplicate_agreement(rows))
        n = a.n.get(name, SETS[name][3])
        todo = [r for r in sample(rows, n, a.seed) if (name, r["row"]) not in done]
        print(f"== {name}: {len(rows)} rows, sample {n}, {len(todo)} to do", flush=True)
        if not todo:
            continue
        crops = Crops(SETS[name][2])
        t0, k = time.time(), 0
        get = lambda r: (r, crops.fetch(r["o"]), crops.fetch(r["a"]), crops.fetch(r["b"]))  # noqa: E731
        with ThreadPoolExecutor(a.threads) as pool:
            for fut in bounded(pool, get, todo, window=4 * a.threads):  # caps decoded crops held in RAM
                try:
                    r, o, xa, xb = fut.result()
                except Exception as e:  # noqa: BLE001  a missing/broken crop costs one row, not the run
                    print("skip row:", e, flush=True)
                    continue
                if not (o.shape == xa.shape == xb.shape):
                    continue
                fa, fb = feat(o, xa), feat(o, xb)
                w.writerow(dict(set=name, row=r["row"], ans=r["ans"], **{f"{k}_a": v for k, v in fa.items()},
                                **{f"{k}_b": v for k, v in fb.items()}))
                k += 1
                if k % 200 == 0:
                    fh.flush()
                    dt = time.time() - t0
                    print(f"  {name} {k}/{len(todo)}  {k / dt:.1f} rows/s  {crops.bytes / dt / 1e6:.0f} MB/s", flush=True)
        fh.flush()
    fh.close()
    (out / "ratings_meta.json").write_text(json.dumps(meta, indent=1))


def add_features(a):
    """Compute the features that features.csv lacks (e.g. wd3/wd5 for a v0 run) for the very same questions, so
    an upgraded Q-hat is fitted on identical data. Progress goes to features_add.csv (resumable); the columns are
    merged into features.csv at the end."""
    from clic27.eval import metrics as M
    out = Path(a.out)
    path, extra = out / "features.csv", out / "features_add.csv"
    rows = M.read_csv(path)
    dev = M.device()
    if dev == "cpu" and not a.allow_cpu:
        raise SystemExit("add needs a GPU (or --allow-cpu)")
    feat = Features(dev)
    missing = [k for k in feat.names() if f"{k}_a" not in rows[0]]
    if not missing:
        print("features.csv already has", feat.names(), flush=True)
        return
    print("adding", missing, "to", len(rows), "questions", flush=True)
    done = {(r["set"], r["row"]) for r in M.read_csv(extra)} if extra.exists() else set()
    cols = ["set", "row"] + [f"{k}_{s}" for s in "ab" for k in missing]
    fh = open(extra, "a", newline="")
    w = csv.DictWriter(fh, cols)
    if not done:
        w.writeheader()
    for name in sorted({r["set"] for r in rows}):
        q = questions(load_answers(name, out / "ratings"))
        todo = [(r, q[int(r["row"])]) for r in rows if r["set"] == name and (name, r["row"]) not in done]
        print(f"== {name}: {len(todo)} to do", flush=True)
        if not todo:
            continue
        crops = Crops(SETS[name][2])
        t0, k = time.time(), 0
        get = lambda t: (t[0], crops.fetch(t[1]["o"]), crops.fetch(t[1]["a"]), crops.fetch(t[1]["b"]))  # noqa: E731
        with ThreadPoolExecutor(a.threads) as pool:
            for fut in bounded(pool, get, todo, window=4 * a.threads):
                try:
                    r, o, xa, xb = fut.result()
                except Exception as e:  # noqa: BLE001  a missing/broken crop costs one row, not the run
                    print("skip row:", e, flush=True)
                    continue
                if not (o.shape == xa.shape == xb.shape):
                    continue
                fa, fb = feat(o, xa, missing), feat(o, xb, missing)
                w.writerow(dict(set=name, row=r["row"], **{f"{k}_a": v for k, v in fa.items()},
                                **{f"{k}_b": v for k, v in fb.items()}))
                k += 1
                if k % 200 == 0:
                    fh.flush()
                    print(f"  {name} {k}/{len(todo)}  {k / (time.time() - t0):.1f} rows/s", flush=True)
        fh.flush()
    fh.close()
    add = {(r["set"], r["row"]): r for r in M.read_csv(extra)}
    merged = [dict(r, **{c: add.get((r["set"], r["row"]), {}).get(c, "nan") for c in cols[2:]}) for r in rows]
    M.write_csv(merged, path)
    print(f"features.csv: {sum((r['set'], r['row']) in add for r in rows)}/{len(rows)} rows have {missing}", flush=True)


# ---------------------------------------------------------------- fit (numpy only)
def transform(name, v):
    """Monotone transforms that make the features closer to linear in log-odds."""
    if name == "msssim":
        return -math.log10(max(1 - v, 1e-6))  # dB-like
    if name in ("lpips_alex", "lpips_vgg", "dists"):
        return -v
    if name in WD:
        return -10 * math.log10(max(v, 1e-12))  # dB-like, higher = better (Cool-chic wd_db)
    return v  # psnr (dB), clipiqa (higher = better)


def design(rows, names):
    """X = phi(A) - phi(B) per row, y = 1 if A preferred. Rows with NaN features are dropped."""
    X, y, s = [], [], []
    for r in rows:
        try:
            d = [transform(k, float(r[f"{k}_a"])) - transform(k, float(r[f"{k}_b"])) for k in names]
        except (KeyError, ValueError):
            continue
        if all(map(math.isfinite, d)):
            X.append(d)
            y.append(1.0 if r["ans"] in ("0", 0) else 0.0)
            s.append(r["set"])
    return np.array(X), np.array(y), np.array(s)


def logistic(X, y, l2=1e-2, iters=50):
    """L2-regularised logistic regression with intercept, Newton/IRLS. Returns (b, w)."""
    Z = np.hstack([np.ones((len(X), 1)), X])
    beta = np.zeros(Z.shape[1])
    R = l2 * np.eye(Z.shape[1])
    R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ beta))
        g = Z.T @ (p - y) / len(y) + R @ beta
        H = (Z * (p * (1 - p))[:, None]).T @ Z / len(y) + R
        step = np.linalg.solve(H, g)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return beta[0], beta[1:]


def accuracy(X, y, b, w):
    return float((((X @ w + b) > 0) == (y > 0.5)).mean()) if len(y) else float("nan")


def fit(a):
    from clic27.eval import metrics as M
    out = Path(a.out)
    rows = M.read_csv(out / "features.csv")
    meta = json.loads((out / "ratings_meta.json").read_text()) if (out / "ratings_meta.json").exists() else {}
    names = [k for k in FULL_REF + NO_REF if f"{k}_a" in rows[0]]
    X, y, s = design(rows, names)
    sd = X.std(0) + 1e-12  # differences are ~zero-mean; scale only
    Xs = X / sd
    sets = sorted(set(s))
    rng = np.random.default_rng(a.seed)
    fold = rng.integers(0, 5, len(y))

    def cv(cols, mask):
        """5-fold CV accuracy of a model on feature columns `cols`, restricted to rows in mask."""
        acc = []
        for k in range(5):
            tr, te = mask & (fold != k), mask & (fold == k)
            b, w = logistic(Xs[tr][:, cols], y[tr])
            acc.append(accuracy(Xs[te][:, cols], y[te], b, w) * te.sum())
        return sum(acc) / mask.sum()

    def single(j, mask):  # sign of one metric difference, no fitting
        return float(((Xs[mask][:, j] > 0) == (y[mask] > 0.5)).mean())

    allc = list(range(len(names)))
    table = []
    for name in sets + ["all"]:
        m = np.ones(len(y), bool) if name == "all" else (s == name)
        row = dict(set=name, n=int(m.sum()))
        row.update({f"sign_{k}": single(j, m) for j, k in enumerate(names)})
        row["qhat_cv"] = cv(allc, m)
        table.append(row)
    # the honest test: train on older years, test on CLIC 2024 (closest to today's codecs)
    cross = {}
    if "2024t" in sets and len(sets) > 1:
        tr, te = s != "2024t", s == "2024t"
        b, w = logistic(Xs[tr], y[tr])
        cross = dict(train=[x for x in sets if x != "2024t"], test="2024t", acc=accuracy(Xs[te], y[te], b, w))
        base = [j for j, k in enumerate(names) if k not in WD]
        if len(base) < len(names):  # the same test without the Wasserstein features (the v0 feature set)
            b0, w0 = logistic(Xs[tr][:, base], y[tr])
            cross["acc_without_wd"] = accuracy(Xs[te][:, base], y[te], b0, w0)
            for k in WD:
                if k in names:
                    j = names.index(k)
                    cross[f"acc_only_{k}"] = float(((Xs[te][:, j] > 0) == (y[te] > 0.5)).mean())
    b, w = logistic(Xs, y)
    model = dict(version=a.version, features=names, transform="see qhat.transform", scale=sd.tolist(),
                 bias=float(b), weights=(w / sd).tolist(), weights_std=w.tolist(), n=int(len(y)),
                 note="Q-hat(O, X) = sum_k weights[k] * transform_k(phi_k(O, X)); P(A > B) = sigmoid(bias + Qa - Qb)")
    (out / f"{a.version}.json").write_text(json.dumps(model, indent=1))
    (out / "fit.json").write_text(json.dumps(dict(table=table, cross=cross, meta=meta), indent=1))
    lines = [f"# Q̂ ({a.version}): dự đoán lựa chọn của người chấm CLIC", "",
             f"{len(y)} câu hỏi có đủ thước đo; thước đo: {', '.join(names)}.", "",
             "Độ chính xác (tỉ lệ đoán đúng ảnh được chọn). `sign_*`: chỉ dựa vào dấu hiệu số của một thước đo; "
             "`qhat_cv`: hồi quy logistic trên mọi thước đo, kiểm định chéo 5 phần.", "",
             "| tập | n | " + " | ".join(f"sign_{k}" for k in names) + " | **qhat_cv** |",
             "|---|---|" + "---|" * (len(names) + 1)]
    for r in table:
        lines.append(f"| {r['set']} | {r['n']} | " + " | ".join(f"{r[f'sign_{k}']:.3f}" for k in names)
                     + f" | **{r['qhat_cv']:.3f}** |")
    if cross:
        lines += ["", f"**Train trên {', '.join(cross['train'])}, test trên 2024t: {cross['acc']:.3f}** "
                      "(con số đáng tin nhất cho CLIC 2027)."]
        if "acc_without_wd" in cross:
            only = ", ".join(f"{k} {cross['acc_only_' + k]:.3f}" for k in WD if "acc_only_" + k in cross)
            lines += ["", f"Cùng phép thử nhưng bỏ hai đặc trưng Wasserstein (bộ đặc trưng v0): "
                          f"{cross['acc_without_wd']:.3f}. Chỉ dùng dấu của một đặc trưng: {only}."]
    if meta:
        lines += ["", "| tập | số câu | chọn đúng ảnh gốc (câu kiểm tra) | câu hỏi lặp lại: tỉ lệ hai người đồng ý |",
                  "|---|---|---|---|"]
        for k, v in meta.items():
            lines.append(f"| {k} | {v['rows']} | {v['sanity']['picked_original']:.3f} (n={v['sanity']['n']}) | "
                         f"{v['duplicates']['agreement']:.3f} (n={v['duplicates']['pairs']}) |")
        lines += ["", "Tỉ lệ đồng ý giữa hai người chấm là trần thực tế của mọi Q̂."]
    lines += ["", "Trọng số chuẩn hóa (độ lớn so sánh được giữa các thước đo):", "",
              "| thước đo | trọng số |", "|---|---|"]
    lines += [f"| {k} | {v:+.3f} |" for k, v in zip(names, w)]
    lines += ["", f"Bias (thiên lệch vị trí A/B): {b:+.3f}. Model: `{a.version}.json`."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.buffer.write(("\n".join(lines) + "\n").encode("utf-8"))  # Windows consoles are not UTF-8


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["features", "add", "fit", "all"])
    ap.add_argument("--out", default="work/qhat")
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    ap.add_argument("--n", nargs="*", default=[], help="per-set sample sizes, e.g. 2024t=8000 2022t=2000")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--version", default="qhat_v1", help="model name: <out>/<version>.json")
    a = ap.parse_args(argv)
    a.n = {k: int(v) for k, v in (x.split("=") for x in a.n)}
    if a.step in ("features", "all"):
        features(a)
    if a.step in ("add", "all"):
        add_features(a)
    if a.step in ("fit", "all"):
        fit(a)


if __name__ == "__main__":
    main()

"""GC codec tests (need torch; the C rANS is built on first use if a compiler exists).

    python -m pytest tests/test_gc.py -q      # or: python tests/test_gc.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402

from clic27.entropy import rans, tables  # noqa: E402
from clic27.gc import alloc, codec, config, rans_c  # noqa: E402
from clic27.gc.model import GCodec  # noqa: E402

TINY = config.merge(config.DEFAULTS["model"], dict(N=32, M=48, Z=32, slices=[8, 8, 32], hyper_hidden=32,
                                                   ctx_hidden=24, gs_width=32, gs_blocks=[1, 1, 1, 1], ga_blocks=1))


def test_c_rans_matches_numpy():
    if rans_c.lib() is None:
        print("C rANS unavailable: skipped")
        return
    cdf, half = tables.gaussian_tables()
    rng = np.random.default_rng(1)
    for n, lanes in itertools.product([0, 1, 5, 1000, 30001], [1, 3, 64]):
        t = rng.integers(0, cdf.shape[0], n)
        s = np.array([rng.integers(0, 2 * half[i] + 2) for i in t], np.int64)
        a = rans.encode(s, t, cdf, n_lanes=lanes)
        b = rans_c.encode(s, t, cdf, lanes=lanes)
        assert a == b, (n, lanes)
        assert np.array_equal(rans_c.decode(b, t, cdf, lanes=lanes), s)


def test_value_coding_escapes():
    cdf, half = tables.gaussian_tables()
    rng = np.random.default_rng(2)
    t = rng.integers(0, 4, 5000)
    q = rng.integers(-40, 40, 5000)  # small scales -> many escapes
    blob = rans_c.encode_values(q, t, cdf, half)
    assert np.array_equal(rans_c.decode_values(blob, t, cdf, half), q)


def test_qmap_table():
    for R in (1, 4, 6):
        c = codec.qmap_cdf(R)
        f = np.diff(c[0].astype(np.int64))
        assert c[0, -1] == rans.L and (f[: 2 * R + 1] >= 1).all()


def _model():
    torch.manual_seed(0)
    m = GCodec(TINY)
    m.check_ranges()
    with torch.no_grad():  # make the integer nets non-trivial
        for p in list(m.h_s.parameters()) + list(m.ctx_a.parameters()) + list(m.ctx_n.parameters()):
            p.normal_(0, 0.2)
    m.update_tables()
    return m.eval()


def test_roundtrip_cpu_and_cuda():
    m = _model()
    x = torch.rand(1, 3, 150, 197)
    ch, cw = m.cell_shape(192, 256)
    cells = torch.randint(-2, 3, (1, 1, ch, cw))
    for level, ent in itertools.product([0, 5], ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])):
        buf, y_enc = codec.compress(m, x, level, 7, cells, ent_dev=ent, return_recon_latent=True)
        y_dec, hd, c2 = codec.decompress_latent(m, buf, ent_dev="cpu")  # decode on CPU whatever encoded
        assert torch.equal(y_enc, y_dec), (level, ent)
        assert torch.equal(c2, cells) and hd["H"] == 150 and hd["W"] == 197 and hd["beta"] == 7
        img = codec.decompress(m, buf, dev="cpu", ent_dev="cpu", synth_dtype=torch.float32)
        assert img.shape == (150, 197, 3)


def test_exact_f32_equals_f64():
    from clic27.entropy.intnet import exact_f32_context
    m = _model()
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    lv = torch.tensor([3])
    H = TINY["hyper_hidden"]
    x = torch.cat([torch.randint(0, 2 ** 16, (1, H, 9, 11)), torch.randint(-32752, 32753, (1, 8 + 1, 9, 11))], 1)
    ref = m.ctx_a[1].forward_int(x.double(), lv)  # float64 path on CPU
    for dev in devs:
        mm = m.to(dev)
        with exact_f32_context():
            got = mm.ctx_a[1].forward_int(x.double().to(dev), lv).cpu()
        assert torch.equal(ref, got), dev
    m.to("cpu")
    # a stream made with the f64 path decodes with the f32 path (and vice versa)
    xi = torch.rand(1, 3, 128, 192)
    codec.EXACT["mode"] = "f64"
    buf, y1 = codec.compress(m, xi, 4, 0, None, return_recon_latent=True)
    codec.EXACT["mode"] = "f32"
    y2, _, _ = codec.decompress_latent(m, buf, ent_dev=devs[-1])
    assert torch.equal(y1, y2)


def test_estimate_close_to_actual():
    m = _model()
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        o = m(x, torch.tensor([3]), torch.zeros(1), torch.zeros(1, 1, 1, 1, dtype=torch.long), noise=False)
    buf = codec.compress(m, x, 3, 0, None)
    est = float(o["bits"]) / 8
    assert abs(len(buf) - est) < 0.15 * est + 64, (len(buf), est)


def test_knapsack_bruteforce():
    rng = np.random.default_rng(3)
    cands = [[alloc.Cand(0, 0, np.zeros(1), float(rng.uniform(100, 1000)), np.zeros(1), float(rng.uniform(0, 1)))
              for _ in range(4)] for _ in range(4)]
    budget = 2000.0
    best = min((sum(cands[i][j].obj for i, j in enumerate(p)), p)
               for p in itertools.product(range(4), repeat=4)
               if sum(cands[i][j].est_bytes for i, j in enumerate(p)) <= budget * 0.99)
    picks = alloc.knapsack(cands, budget, 0.0, units=20000)
    got = sum(cands[i][j].obj for i, j in enumerate(picks))
    assert sum(cands[i][j].est_bytes for i, j in enumerate(picks)) <= budget
    assert got <= best[0] + 1e-9


def test_reward_roundtrip_keeps_vgg(tmp_path=None):
    """reward.pt must score exactly as the fitted model (regression: VGG used to be dropped and re-made random)."""
    import tempfile
    from clic27.gc.reward import PrefDistance, load_reward
    torch.manual_seed(0)
    m = PrefDistance(pretrained=False).eval()  # random VGG stands in for ImageNet weights; no download
    with torch.no_grad():
        for p in m.w_raw:
            p.normal_()
    o, x = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "reward.pt"
        torch.save(dict(state_dict=m.state_dict(), report={}), path)
        r = load_reward(path)
    with torch.no_grad():
        assert torch.equal(m(o, x), r(o, x))


def test_data_paths_fail_loudly_and_patches_track_sources():
    """A missing/empty training source is an error; patches cut from other sources are never reused silently."""
    import json
    import tempfile
    from PIL import Image
    from clic27.gc.data import list_images, patches_ready, prep_patches
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        a, b, empty = d / "a", d / "b", d / "empty"
        for folder in (a, b):
            folder.mkdir()
            Image.new("RGB", (600, 600), (10, 20, 30)).save(folder / "x.png")
        empty.mkdir()
        assert len(list_images([a, d / "missing"])) == 1  # non-strict keeps the old behaviour
        for bad in ([a, d / "missing"], [a, empty]):
            try:
                list_images(bad, strict=True)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError(f"strict listing accepted {bad}")
        out = d / "patches"
        prep_patches([a], out, per_image=1, size=512, workers=1)
        assert json.loads((out / ".done").read_text())["patches"] == 1
        assert patches_ready(out, [a])
        try:
            patches_ready(out, [a, b])  # data changed after the cut
        except RuntimeError:
            pass
        else:
            raise AssertionError("stale patches accepted")


def _tiny_train_cfg(d: Path):
    from PIL import Image
    imgs = d / "imgs"
    imgs.mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(imgs / f"{i}.png")
    cfg = config.merge(config.DEFAULTS, dict(
        model=TINY,
        data=dict(train_dirs=[str(imgs)], crop=64, min_side=64, workers=0, valid_dir=""),
        stages=dict(rd=dict(steps=4, batch=2, lpips=False), gan=dict(steps=2, batch=2, lpips=False)),
        train=dict(amp="none", log_every=1, ckpt_every=100, eval_every=100)))
    return cfg


def test_stage_stop_resume_and_handoff():
    """Signal / time-budget stops checkpoint and exit 75 without model.pt; rerun resumes; a stage never starts
    from an unfinished one; a wrong --init fails loudly."""
    import json
    import tempfile
    from clic27.gc import train
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cfg, work = _tiny_train_cfg(d), d / "train"

        def expect_partial(**kw):
            try:
                train.run_stage(cfg, work, "rd", **kw)
            except SystemExit as e:
                assert e.code == train.EXIT_PARTIAL, e.code
            else:
                raise AssertionError("stage should have stopped early")
            assert (work / "rd" / "last.pt").exists() and (work / "rd" / "partial.json").exists()
            assert not (work / "rd" / "model.pt").exists()

        expect_partial(max_minutes=1e-9)                       # time budget
        assert json.loads((work / "rd" / "partial.json").read_text())["reason"] == "max_minutes"
        try:                                                   # gan must not start from the unfinished rd
            train.run_stage(cfg, work, "gan")
        except SystemExit as e:
            assert "unfinished" in str(e.code), e.code
        else:
            raise AssertionError("gan started although rd is unfinished")
        assert not (work / "gan").exists() or not (work / "gan" / "model.pt").exists()
        train._Stop.signum = 15                                # preemption signal
        try:
            expect_partial()
        finally:
            train._Stop.signum = 0
        assert json.loads((work / "rd" / "partial.json").read_text())["reason"] == "signal"
        train.run_stage(cfg, work, "rd")                       # resume to the end
        sd = torch.load(work / "rd" / "model.pt", map_location="cpu", weights_only=False)
        assert sd["complete"] and sd["step"] == 4 and not (work / "rd" / "partial.json").exists()
        train.run_stage(cfg, work, "gan")                      # previous stage complete: accepted
        assert torch.load(work / "gan" / "model.pt", map_location="cpu", weights_only=False)["complete"]
        try:
            train.run_stage(cfg, d / "other", "rd", init=str(d / "nope.pt"))
        except SystemExit as e:
            assert "does not exist" in str(e.code)
        else:
            raise AssertionError("missing --init accepted")


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)

"""CPU/numpy-only checks for ratflow.eval.metrics (no torch needed).

    pytest            # or: python tests/test_metrics.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ratflow.eval import metrics as M  # noqa: E402


def test_psnr_and_pooling():
    a = np.zeros((4, 4, 3), np.uint8)
    b = a + 1
    assert M.mse(a, b) == 1.0
    assert abs(M.psnr(a, b) - 48.1308) < 1e-3
    # pooled PSNR weights MSE by pixel count, not PSNR by image
    assert abs(M.pooled_psnr([1.0, 4.0], [3, 1]) - M.psnr_from_mse(1.75)) < 1e-12


def test_patches_hifc_protocol():
    ps = list(M.patches(np.zeros((1356, 2040, 3), np.uint8)))
    assert len(ps) == 5 * 7 + 4 * 7  # grid + grid shifted by 128
    assert all(p.shape == (256, 256, 3) for p in ps)


def test_images_skips_macos_debris(tmp_path):
    (tmp_path / "__MACOSX").mkdir()
    for p in ["__MACOSX/x.png", "._a.png", "b.png", "c.JPG", "d.txt"]:
        (tmp_path / p).touch()
    assert [p.name for p in M.images(tmp_path)] == ["b.png", "c.JPG"]


def test_text_helpers():
    assert M.edit_distance("kitten", "sitting") == 3
    assert M.cer(["abcd"], ["abxd"]) == 0.25
    assert np.isnan(M.cer([""], ["x"]))
    assert M.box_xyxy([[1, 2], [30, 2], [30, 9], [1, 9]], 100, 100) == (0, 0, 34, 13)
    a = np.zeros((10, 10, 3), np.uint8)
    b = a.copy()
    b[0, 0] = 10
    assert np.isnan(M.region_psnr(a, b, []))
    assert M.region_psnr(a, b, [(5, 5, 10, 10)]) == M.psnr_from_mse(1e-10)


if __name__ == "__main__":
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(Path(tempfile.mkdtemp())) if "tmp_path" in fn.__code__.co_varnames else fn()
            print("ok", name)

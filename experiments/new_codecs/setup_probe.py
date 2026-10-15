"""Download official AFP-GIC/DC-VIC weights and check imports; CPU only."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
root = Path.home() / "clic/research_20261015"
vendor = root / "vendor"
sys.path.insert(0, str(vendor))
packages = {"addict": "addict==2.4.0", "python_log_indenter": "python-log-indenter==0.9",
            "omegaconf": "omegaconf", "einops": "einops", "timm": "timm"}
missing = [pkg for module, pkg in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(vendor), *missing], check=True)
from huggingface_hub import hf_hub_download
print("AFP", hf_hub_download("yifeipet/AFP-GIC", "afp_gic_release.pth.tar", local_dir=root / "weights"), flush=True)
path = root / "weights/dc_vic_patchgan.pth.tar"
if not path.exists():
    subprocess.run(["curl", "--fail", "--location", "--retry", "2", "--max-time", "600",
                    "https://drive.usercontent.google.com/download?id=1GmrJ-4IWsQTSZIMLHY6i3nD_VKg_A3-_&confirm=xxx",
                    "-o", str(path) + ".partial"], check=True)
    temp = Path(str(path) + ".partial")
    if temp.stat().st_size < 1000000:
        raise RuntimeError("DC-VIC download is too small; may be an HTML error")
    temp.rename(path)
print("DCVIC", path, path.stat().st_size, flush=True)
sys.path.insert(0, str(root / "AFP_GIC/public_release/runtime"))
import eval_public_release
print("AFP runtime import OK", flush=True)

"""Load diffusers-format checkpoints without diffusers.

Sources, in order of preference:
- a local directory containing config.json and diffusion_pytorch_model{.safetensors,-0000x-of-0000y.safetensors,.pt}
- a Hugging Face repo id plus subfolder (needs huggingface_hub; used for training, not in the CLIC decoder)

The CLIC server image has torch but no safetensors, so `save_pt` writes a plain torch file for the decoder zip.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

WEIGHT_NAMES = ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.safetensors.index.json",
                "diffusion_pytorch_model.pt", "model.pt")


def resolve(repo_or_dir: str, subfolder: str | None = None) -> Path:
    """Local directory with config + weights; downloads from the Hub if repo_or_dir is not a directory."""
    p = Path(repo_or_dir) / (subfolder or "")
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    pattern = f"{subfolder}/*" if subfolder else "*"
    root = snapshot_download(repo_or_dir, allow_patterns=[pattern])
    return Path(root) / (subfolder or "")


def load_config(d: Path) -> dict:
    return json.loads((Path(d) / "config.json").read_text())


def load_state_dict(d: Path) -> dict:
    d = Path(d)
    for name in WEIGHT_NAMES:
        f = d / name
        if not f.exists():
            continue
        if name.endswith(".index.json"):
            from safetensors.torch import load_file
            shards = sorted(set(json.loads(f.read_text())["weight_map"].values()))
            sd = {}
            for s in shards:
                sd.update(load_file(str(d / s)))
            return sd
        if name.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(str(f))
        return torch.load(f, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"no weights in {d} (looked for {WEIGHT_NAMES})")


def save_pt(model: torch.nn.Module, config: dict, d: Path, dtype: torch.dtype | None = None):
    """Write config.json + model.pt (for environments without safetensors)."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    sd = {k: (v.to(dtype) if dtype is not None and v.is_floating_point() else v).contiguous()
          for k, v in model.state_dict().items()}
    torch.save(sd, d / "model.pt")
    (d / "config.json").write_text(json.dumps(config, indent=1))

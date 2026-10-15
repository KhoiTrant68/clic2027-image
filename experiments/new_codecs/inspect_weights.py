from pathlib import Path
import torch

root = Path.home() / "clic/research_20261015"
for path in sorted((root / "weights").glob("*.pth.tar")):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    print(path.name, list(ck), flush=True)
    state = ck.get("comp_model", {})
    print("prefixes", sorted(set(k.removeprefix("module.").split(".")[0] for k in state)), flush=True)
    print("tensors", len(state), flush=True)

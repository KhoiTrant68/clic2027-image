"""Collect small, auditable artifacts; leave weights, images and streams on server."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path.home() / "clic/research_20261015"
    runs = ["pilot_afp_strict", "pilot_dcvic_strict", "pilot_pulse_base",
            "pilot_pulse_large", "full_pulse_large"]
    manifest = {"root": str(root), "runs": {}, "repos": {}, "weights": {},
                "invalid_excluded": ["smoke_afp: non-strict checkpoint load; entropy parameters skipped"],
                "latest_user_vram_limit_bytes": 30_000_000_000,
                "these_runs_used_stricter_half_device_guard": True}
    for name in ("AFP_GIC", "DC_VIC", "GenCodec", "ml-pico"):
        manifest["repos"][name] = subprocess.check_output(
            ["git", "-C", str(root / name), "rev-parse", "HEAD"], text=True).strip()
    for p in sorted((root / "weights").rglob("*")):
        if p.is_file() and not p.name.endswith(".partial"):
            manifest["weights"][str(p.relative_to(root))] = dict(bytes=p.stat().st_size, sha256=sha(p))
    for name in runs:
        d = root / name
        points = [json.loads(p.read_text()) for p in sorted(d.glob("q*/*.json"))]
        good = [r for r in points if r["status"] == "ok"]
        memory = [json.loads(s)["used_mib"] for s in (d / "vram.jsonl").read_text().splitlines()]
        manifest["runs"][name] = dict(
            count=len(points), success=len(good), images=len({r["name"] for r in good}),
            max_device_used_mib=max(memory), max_allocated_mib=max(r["peak_allocated_mib"] for r in good),
            decode_median_s=statistics.median(r["decode_s"] for r in good),
            decode_min_s=min(r["decode_s"] for r in good), decode_max_s=max(r["decode_s"] for r in good),
            meta=json.loads((d / "meta.json").read_text()))
    sources = [Path.home() / "clic/inputs/qhat_v0/qhat/qhat_v0.json",
               Path.home() / "clic/work_merge_1013_v0/metrics.csv",
               Path.home() / "clic/clic2027-image/src/clic27/eval/metrics.py",
               Path.home() / "clic/clic2027-image/experiments/qhat/qhat.py",
               Path.home() / "clic/clic2027-image/experiments/bakeoff/bakeoff.py"]
    manifest["source_sha256"] = {str(p): sha(p) for p in sources}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with tarfile.open(root / "results_light.tar.gz", "w:gz") as tar:
        tar.add(root / "manifest.json", arcname="manifest.json")
        tar.add(sources[0], arcname="qhat_v0.json")
        for name in [*runs, "pilot_scores", "full_scores", "fresh_decode"]:
            for p in sorted((root / name).rglob("*")):
                if p.is_file() and p.suffix in {".json", ".jsonl", ".csv"}:
                    tar.add(p, arcname=str(p.relative_to(root)))
        for p in sorted(root.glob("*.log")):
            if p.name.startswith(("pilot_", "full_")):
                tar.add(p, arcname="logs/" + p.name)
    print(json.dumps(manifest["runs"], indent=2))


if __name__ == "__main__":
    main()

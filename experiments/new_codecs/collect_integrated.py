"""Validate integrated results and collect compact deployment evidence."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile


def read_csv(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def main():
    root = Path.home() / "clic/research_20261015"
    work = root / "integrated_bakeoff"
    points = read_csv(work / "points/pulse.csv")
    sizes = json.loads((work / "budgets.json").read_text())["sizes"]
    assert {(r["key"], r["name"]) for r in points} == {(f"q{q}", n) for q in range(5) for n in sizes}
    identical_prior = 0
    for r in points:
        p = work / "streams/pulse" / r["key"] / (r["name"] + ".bin")
        data = p.read_bytes()
        assert data[:4] == b"PL\x01\x00" and len(data) == int(r["bytes"]) == int(r["bytes_base"]) + 4
        prior = root / "full_pulse_large" / r["key"] / (r["name"] + ".pulse")
        if prior.exists():
            assert data[4:] == prior.read_bytes(), f"Payload changed: {p}"
            identical_prior += 1
    results = dict(pulse_points=len(points), identical_prior_payloads=identical_prior,
                   transmitted_mode_header_bytes=4, offline={})
    for version in (1, 2):
        d = root / f"offline_decode_v{version}"
        benchmark = json.loads((d / "benchmark.json").read_text())
        profile = json.loads((d / "decode.json").read_text())
        assert benchmark["output_count"] == len(sizes) and all(c["equal"] for c in benchmark["checks"])
        results["offline"][f"v{version}"] = dict(
            total_wall_s=benchmark["total_wall_s"], unpack_s=benchmark["unpack_s"],
            process_wall_s=benchmark["process_wall_s"], load_s=profile["load_s"],
            decode_sum_s=sum(r["decode_s"] for r in profile["images"]),
            png_write_sum_s=sum(r["write_s"] for r in profile["images"]),
            peak_rss_kib=profile["peak_rss_kib"], matching_images=benchmark["output_count"],
            max_device_used_mib=max(json.loads(s)["used_mib"] for s in (d / "decode.vram.jsonl").read_text().splitlines()))
    results["integrated_max_device_used_mib"] = max(
        json.loads(s)["used_mib"] for s in (work / "vram.jsonl").read_text().splitlines())
    summary = json.loads((work / "summary.json").read_text())
    results["quality"] = {k: v for k, v in summary.items() if k.endswith("/all") and k.startswith(("mix@", "pulse@"))}
    (root / "integration_checks.json").write_text(json.dumps(results, indent=2))
    with tarfile.open(root / "integration_results.tar.gz", "w:gz") as tar:
        for name in ("integration_checks.json", "integrated_run.log", "package_v1.json", "package_v2.json"):
            tar.add(root / name, arcname=name)
        for d in (work, root / "offline_decode_v1", root / "offline_decode_v2"):
            for p in sorted(d.rglob("*")):
                if p.is_file() and p.suffix in {".json", ".jsonl", ".csv", ".log", ".md"} and "decoder" not in p.relative_to(d).parts:
                    tar.add(p, arcname=p.relative_to(root).as_posix())
        source_hashes = {}
        for name in ("experiments/bakeoff/bakeoff.py", "experiments/bakeoff/pulse_codec.py",
                     "experiments/new_codecs/pulse_decode.py", "experiments/new_codecs/benchmark_pulse.py",
                     "experiments/new_codecs/package_pulse.py"):
            source_hashes[name] = hashlib.sha256((root / "integration_repo" / name).read_bytes()).hexdigest()
        (root / "integration_source_hashes.json").write_text(json.dumps(source_hashes, indent=2))
        tar.add(root / "integration_source_hashes.json", arcname="source_hashes.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

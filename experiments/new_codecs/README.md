# Public-model screening, 2026-10-15

Actual bitstream probes for AFP-GIC, DC-VIC and PULSE. These are small research
scripts for the existing `guess` server, not a new evaluation framework.

Results: `results/2026-10-15_new_codecs/`; interpretation and next experiments:
`reports/CLIC2027_new_models_2026-10-15.md`.

## Environment and checkpoints

Remote root: `~/clic/research_20261015`. Model checkout SHAs and all weight hashes
are in the results manifest. Keep public runtimes/checkpoints there, separate
from the dirty remote project checkout. `setup_probe.py` prepares the AFP/DC-VIC
minimal imports and public weights. It is an initial setup helper, not a frozen
dependency installer: check the manifest hashes when repeating the experiment.

AFP/DC-VIC: `~/clic/venv/bin/python`, Torch 2.11, CompressAI 1.2.8. The probe
renames old entropy ParameterList keys and requires strict checkpoint loading.
DC-VIC's full checkpoint supplies its VQ model; do not substitute random VQGAN.

PULSE: `~/clic/clic27_l4/venv/bin/python`, Torch 2.6. Build the pinned GenCodec
`PULSE` package (CPU rANS extension) into `pulse_vendor` with `pip install
--no-deps --no-build-isolation --target ... .`. Run its `download_models.py` for
`pulse-perceptual pulse-l-perceptual`; its manifest checks download hashes.
Keep `config.json`, `model.pt`, `entropy_control_int.pt` together.

## Run

Select one physical GPU and do not overlap jobs on that GPU. Examples on server:

```bash
cd ~/clic/research_20261015
CUDA_VISIBLE_DEVICES=0 ~/clic/clic27_l4/venv/bin/python probe_pulse.py \
  --model pulse-l-perceptual --out full_pulse_large --qualities 0 1 2 4
CUDA_VISIBLE_DEVICES=1 ~/clic/venv/bin/python score.py \
  --repo ~/clic/clic2027-image --baseline ~/clic/work_merge_1013_v0 \
  --qhat ~/clic/inputs/qhat_v0/qhat/qhat_v0.json \
  --data ~/clic/work_full_gpu/bakeoff/valid \
  --out full_scores --runs full_pulse_large
CUDA_VISIBLE_DEVICES=0 ~/clic/clic27_l4/venv/bin/python verify_pulse.py
```

Use `probe.py --model afp` or `--model dcvic` with the main venv. Both probes
support `--only` filename prefixes for pilots. `score.py` also accepts these
prefixes; otherwise it requires all images. It compares with existing baseline
points using the repo Qhat scorer and byte allocator. It does not train models.

Output directories are immutable experiment identities: resume only the same
code/checkpoint/config. Use a new directory after changing any of these. An
existing successful JSON is skipped. Do not mix partial or old invalid outputs.

## VRAM policy

The user's latest allowance is **30 GB/device**. These completed tests retain
the stricter `half_vram.py` guard: half the reported device memory, at most
18 GiB PyTorch allocator, extra context/service headroom, whole-device polling
every 0.25 s, fail closed if monitoring fails. This is not a hardware partition.
Do not remove it merely because more memory is allowed. Future training that
needs a larger cap must configure and validate a 30 GB total-device guard first.

`half_vram.py` has CPU-only budget checks; the probe scripts passed compilation
on server. All 204 valid points wrote/read real streams with full image shapes.
Three PULSE-L streams also passed fresh-process same-pixel decoding. This is not
yet cross-platform parity, official L4 benchmarking or a competition package.

`collect.py` creates a small result archive including per-point metadata, score
tables, memory traces and SHA256 provenance. PNGs, streams and large weights stay
on server. An earlier AFP non-strict-load smoke is explicitly invalid/excluded.

## Main bake-off integration and offline deployment

The main CLI now accepts `--cands pulse --pulse-bundle PATH --pulse-runtime PATH`
and `--pulse-q 0 1 2 3 4`. `pulse` is opt-in, so legacy default candidate runs do
not require PULSE. Streams are stored at `streams/pulse/qN/<name>.bin`. Their
first four bytes are `PL`, version 1, flags 0; the remaining payload is the
author's checked PULSE envelope (QP, dimensions, bundle ID included). `bytes`
includes all four transmitted bytes; no second estimated header is added.

Use the integrated runner to separate Torch 2.6 coding from Torch 2.11 scoring:

```bash
CUDA_VISIBLE_DEVICES=0 ~/clic/venv/bin/python experiments/new_codecs/run_integrated.py \
  --gpu-memory-guard --repo "$PWD" \
  --baseline ~/clic/work_merge_1013_v0 --out ~/clic/research_20261015/integrated_repeat \
  --data ~/clic/work_full_gpu/bakeoff/valid \
  --bundle ~/clic/research_20261015/weights/pulse/pulse-l-perceptual \
  --runtime ~/clic/research_20261015/pulse_vendor \
  --qhat ~/clic/inputs/qhat_v0/qhat/qhat_v0.json \
  --codec-python ~/clic/clic27_l4/venv/bin/python \
  --metric-python ~/clic/venv/bin/python
```

The output directory must be new. Baseline points/metrics are copied, not edited.
The runner calls main bake-off prepare/points/metrics/allocate/summary and refuses
an incomplete PULSE grid. Keep reconstructions between `--defer-metrics` and
metrics; they are the input to the scorer. Archive the runtime source as well as
bundle hashes; the bundle identity check does not detect arbitrary runtime edits.

`package_pulse.py --repo ... --runtime ... --bundle ... --license ... --out ...`
creates a relocatable Linux/Python-3.12 decoder directory and ZIP. It includes
weights, codec source, rANS extension, original license, helper scripts and a
checksum manifest. Python, Torch, NumPy, Pillow and system CUDA libraries are
provided by the host environment, not included in the ZIP. This research kit is
not yet an official CLIC submission or a mixed-codec decoder.

`benchmark_pulse.py --gpu-memory-guard --archive ... --bakeoff ... --out ...
--python /absolute/path/to/python` selects the full-corpus PULSE allocation at
0.3 bpp, verifies actual input bytes, unpacks into a new directory and launches
isolated Python. The decoder blocks Python TCP connections and uses an empty HF
cache. This is an offline dependency check, not OS-level network isolation.
It compares all decoded pixels with bake-off reconstructions after timing ends.

Outputs use lossless PNG with `compress_level=0`: same pixels, larger PNG files,
less CPU recompression. Output PNG size is not codec-stream bitrate. Wall time
includes unpack, process startup, checksum validation, model load, decode and PNG
write; comparison with reference images is excluded. Use identical measurement
scope when comparing other decoders. Keep the guard even on this small model.

Completed integrated run: 150 points, 120 payloads exactly match the earlier
probe (before adding the four-byte mode header), two framing unit tests pass,
and 30/30 independent output images match. Results and limits are in
`results/2026-10-15_pulse_integration` and
`reports/CLIC2027_pulse_integration_2026-10-15.md`.

## MS-ILLM selected-stream audit

`verify_msillm.py` audits the five MS-ILLM images selected by the integrated
mix at 0.3 bpp. It creates a new output directory, encodes actual 28-byte framed
streams, launches a fresh decoder with Python TCP blocked and compares pixels.
Explicit local hub source and checkpoint paths are required; this is not yet a
relocatable decoder package. Add `--deterministic` for the configuration that
passed fresh-process parity. The default run exposed one-level pixel differences.

```bash
CUDA_VISIBLE_DEVICES=0 ~/clic/venv/bin/python verify_msillm.py \
  --gpu-memory-guard --deterministic --out msillm-selected-mix03-repeat \
  --hub ~/.cache/torch/hub/facebookresearch_NeuralCompression_main \
  --weights ~/.cache/torch/hub/checkpoints --bakeoff integrated_bakeoff \
  --data ~/clic/work_full_gpu/bakeoff/valid \
  --reference ~/clic/work_full_gpu/bakeoff/recon/msillm
```

Use `--limit 1` for smoke. The decoder receives neither source images nor the
selection table: model quality and dimensions come from its stream. Full run:
5/5 exact fresh-process outputs, +120 bytes versus baseline accounting, still
within budget. Deterministic output differs slightly from old reconstructions;
rescore before claiming the old Qhat value for the new decoder. See
`reports/CLIC2027_msillm_stream_audit_2026-10-15.md` and the matching results folder.

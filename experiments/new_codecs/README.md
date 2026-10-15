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

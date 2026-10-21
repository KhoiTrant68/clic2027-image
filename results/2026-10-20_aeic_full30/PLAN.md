# AEIC full30 — protocol declared before scoring

Date: 20 October 2026, Asia/Bangkok. User authorized full evaluation, visual review and conditional fine-tuning.

## Fixed experiment

| Variant | Checkpoint | Images | Device | Allocator / own-process stop |
|---|---|---:|---|---|
| AEIC-SE | official SE-ft2 | 30 | L40 GPU 0 | 8 / 10 GiB |
| AEIC-ME | official ME-ft2 | 30 | L40 GPU 1 | 8 / 10 GiB |

Use the exact successful pilot recipe: FP32, seed 0, deterministic cuDNN, TF32 off, compilation off, xformers on; latent tile 64, overlap 16, VAE tile 64; color correction using source statistics off. Do not resize input images. Count real stream bytes, including four-byte AEIC envelope. Re-decode all 60 streams in fresh processes and require exact RGB pixels before scoring. Preserve partial failures; never count them as valid points.

Server root: `~/clic/aeic_20261016/full30_20261020`. Parent process IDs at launch: SE 1954043, ME 1954982. Original author commit and weight hashes are in the pilot evidence. Estimated encode plus independent decode: 25–40 minutes with both variants in parallel; resource contention can extend this. JSON status, per-stage logs, per-point records and 0.25-second VRAM monitoring are the experiment tracker.

## Comparison

Reference: `~/clic/research_20261015/integrated_bakeoff`, including all 150 PULSE-L q0–q4 points. Recompute the same baseline allocation and require its Qhat to reproduce the stored mix scores within 1e-6. Then add the 60 AEIC points and allocate under exactly the same whole-corpus budgets 0.075, 0.15 and 0.3 bpp. Record exact bytes, delta Qhat, selected-model counts and every changed image. Do not compare standalone AEIC's ~0.02–0.03 bpp directly with a higher-rate baseline as if rates were matched.

Primary practical gate (a screening threshold, not a statistical-significance claim): at least +0.02 mean Qhat and at least 3/30 AEIC-selected images at one budget. A metric gain alone does not authorize a claim about human preference. Include selected-image decode cost and model weight footprint before deciding whether a gain is useful.

## Visual review

For every distinct changed before/after selection, generate three 512-pixel crops using a fixed seed: one central, two random nonflat source crops. Retain all changed images, including ones whose individual score decreases to save bits. Show the original plus randomized anonymous A/B reconstructions. Keep the answer key out of the HTML page and do not prefill votes. The experiment produces a local review pack, not a completed human study. AI visual observations must be labeled separately from human votes.

## Fine-tuning decision

Proceed only after a meaningful measured gain, satisfactory visual fidelity and acceptable decoder cost. If the metric gate fails, record a no-go and avoid an expensive training run. If it passes but visual evidence or deployment feasibility is unresolved, prepare the minimal training recipe and explicitly record the outstanding gate instead of claiming training completed.

Potential first fine-tune: AEIC-ME initialized from ft2, target near 0.075 bpp via a lower rate-loss weight than the author's lambda=2. Use separate training images (DIV2K train is present on server), never the 30 validation images. Calibrate lambda in a short pilot before a long run; batch 1, checkpointing/accumulation and a measured resource cap. Exact optimizer/config must be verified against author training code before launching. This protocol does not assert that a new training run will be warranted.

# Cool-chic probe, 2026-10-09 (L40 server, run by hand)

Cool-chic 5.0.1 @ a6fe38a, `cc_encode.py --tune wasserstein --lmbda 0.004 --n_itr 2000` (the minimum the intra preset
accepts; default 1e4), image `2684452d` (2048x1360, 2.79 MP), PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.

| | value |
|---|---|
| peak GPU memory | 21.06 GB (does not fit 10 GB/GPU; OOM at 9.4 and 20 GB caps under bakeoff.py) |
| encode time | training 934 s + quantize_model 221 s + RDOQ 682 s, about 31 min |
| rate | 0.1114 bpp (network parameters 0.0067 bpp) |
| decoder | 2390 MAC/pixel (ARM 1178, synthesis 1107) |
| decode time | 14.6 s on GPU, 13.9 s on CPU (cc_decode.py, Python); GPU and CPU outputs byte-identical |

Same image, Q-hat v0 (features computed with experiments/qhat on the server):

| codec | bpp | PSNR | LPIPS | DISTS | Q-hat |
|---|---|---|---|---|---|
| VTM 4:2:0 qp40 | 0.100 | 30.46 | 0.263 | 0.191 | -2.30 |
| Cool-chic wasserstein | 0.111 | 29.11 | 0.109 | 0.091 | -0.41 |
| MS-ILLM, interpolated q2-q3 | 0.111 | ~28.8 | ~0.076 | ~0.078 | ~+0.22 |
| CoD-Lite 0_1250 | 0.129 | 25.86 | 0.082 | 0.071 | +0.58 |

Verdict: not a base for now. ~0.6 Q-hat behind MS-ILLM at equal rate on this image (2000 iterations only, and Q-hat v0
has no Wasserstein feature), decode ~20x over the 20 s / 30 images target with the Python decoder, 21 GB / 31 min
per encode. Worth taking: Wasserstein Distortion as a Q-hat feature; per-image (encoder-side) optimisation applied
to MS-ILLM / CoD-Lite latents.

# Student vs teacher, DiT-IC q3, 30 CLIC validation images

Checkpoint: distill/run_q3_gan01/snap_2000.pt (step 2000). Same bitstreams (mean 0.0762 bpp).

| decoder | PSNR | MS-SSIM | LPIPS | DISTS | WD3 | Q-hat v0 |
|---|---|---|---|---|---|---|
| teacher | 24.703 | 0.8828 | 0.0826 | 0.0692 | 0.2144 | +0.157 |
| student | 24.281 | 0.8787 | 0.1209 | 0.0969 | 0.3847 | -0.639 |

Q-hat v0 student - teacher: mean -0.796, worst -1.460, images worse by > 0.1: 30/30

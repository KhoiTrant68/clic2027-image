# Student vs teacher, DiT-IC q3, 30 CLIC validation images

Checkpoint: distill/run_q3_gan01/snap_18000.pt (step 18000). Same bitstreams (mean 0.0762 bpp).

| decoder | PSNR | MS-SSIM | LPIPS | DISTS | WD3 | Q-hat v0 |
|---|---|---|---|---|---|---|
| teacher | 24.703 | 0.8828 | 0.0826 | 0.0692 | 0.2144 | +0.157 |
| student | 24.656 | 0.8796 | 0.1018 | 0.0819 | 0.2537 | -0.112 |

Q-hat v0 student - teacher: mean -0.268, worst -1.040, images worse by > 0.1: 30/30

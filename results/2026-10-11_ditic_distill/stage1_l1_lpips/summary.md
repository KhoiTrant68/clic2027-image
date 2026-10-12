# Student vs teacher, DiT-IC q3, 30 CLIC validation images

Checkpoint: distill/run_q3/best.pt (step 20000). Same bitstreams (mean 0.0762 bpp).

| decoder | PSNR | MS-SSIM | LPIPS | DISTS | WD3 | Q-hat v0 |
|---|---|---|---|---|---|---|
| teacher | 24.703 | 0.8828 | 0.0826 | 0.0692 | 0.2144 | +0.157 |
| student | 25.247 | 0.8849 | 0.1108 | 0.0839 | 0.2634 | -0.269 |

Q-hat v0 student - teacher: mean -0.425, worst -0.884, images worse by > 0.1: 30/30

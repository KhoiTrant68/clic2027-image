# Student vs teacher, DiT-IC q3, 30 CLIC validation images

Checkpoint: distill/run_q3_gan01/snap_30000.pt (step 30000). Same bitstreams (mean 0.0762 bpp).

| decoder | PSNR | MS-SSIM | LPIPS | DISTS | WD3 | Q-hat v0 |
|---|---|---|---|---|---|---|
| teacher | 24.703 | 0.8828 | 0.0826 | 0.0692 | 0.2144 | +0.157 |
| student | 24.765 | 0.8809 | 0.0950 | 0.0765 | 0.2431 | -0.012 |

Q-hat v0 student - teacher: mean -0.168, worst -0.981, images worse by > 0.1: 19/30

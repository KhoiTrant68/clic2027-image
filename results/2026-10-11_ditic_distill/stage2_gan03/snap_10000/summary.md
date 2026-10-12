# Student vs teacher, DiT-IC q3, 30 CLIC validation images

Checkpoint: distill/run_q3_gan03/snap_10000.pt (step 10000). Same bitstreams (mean 0.0762 bpp).

| decoder | PSNR | MS-SSIM | LPIPS | DISTS | WD3 | Q-hat v0 |
|---|---|---|---|---|---|---|
| teacher | 24.703 | 0.8828 | 0.0826 | 0.0692 | 0.2144 | +0.157 |
| student | 24.595 | 0.8760 | 0.1085 | 0.0857 | 0.2825 | -0.240 |

Q-hat v0 student - teacher: mean -0.397, worst -1.374, images worse by > 0.1: 29/30

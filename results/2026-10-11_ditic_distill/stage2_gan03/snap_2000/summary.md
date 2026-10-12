# Student vs teacher, DiT-IC q3, 30 CLIC validation images

Checkpoint: distill/run_q3_gan03/snap_2000.pt (step 2000). Same bitstreams (mean 0.0762 bpp).

| decoder | PSNR | MS-SSIM | LPIPS | DISTS | WD3 | Q-hat v0 |
|---|---|---|---|---|---|---|
| teacher | 24.703 | 0.8828 | 0.0826 | 0.0692 | 0.2144 | +0.157 |
| student | 24.588 | 0.8765 | 0.1321 | 0.0982 | 0.3231 | -0.616 |

Q-hat v0 student - teacher: mean -0.772, worst -1.441, images worse by > 0.1: 30/30

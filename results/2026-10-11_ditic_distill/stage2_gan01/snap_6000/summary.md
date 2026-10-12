# Student vs teacher, DiT-IC q3, 30 CLIC validation images

Checkpoint: distill/run_q3_gan01/snap_6000.pt (step 6000). Same bitstreams (mean 0.0762 bpp).

| decoder | PSNR | MS-SSIM | LPIPS | DISTS | WD3 | Q-hat v0 |
|---|---|---|---|---|---|---|
| teacher | 24.703 | 0.8828 | 0.0826 | 0.0692 | 0.2144 | +0.157 |
| student | 24.819 | 0.8809 | 0.1069 | 0.0847 | 0.2642 | -0.215 |

Q-hat v0 student - teacher: mean -0.371, worst -1.008, images worse by > 0.1: 30/30

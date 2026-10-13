# Student vs teacher, DiT-IC q4, 30 CLIC validation images

Checkpoint: distill/run_q4_gan01/snap_25000.pt (step 25000). Same bitstreams (mean 0.0996 bpp).

| decoder | PSNR | MS-SSIM | LPIPS | DISTS | WD3 | Q-hat v0 |
|---|---|---|---|---|---|---|
| teacher | 24.974 | 0.8883 | 0.0783 | 0.0650 | 0.1952 | +0.272 |
| student | 25.043 | 0.8877 | 0.0925 | 0.0740 | 0.2340 | +0.090 |

Q-hat v0 student - teacher: mean -0.182, worst -0.743, images worse by > 0.1: 23/30

# Student vs teacher, DiT-IC q4, 30 CLIC validation images

Checkpoint: distill/run_q4_gan01/snap_30000.pt (step 30000). Same bitstreams (mean 0.0996 bpp).

| decoder | PSNR | MS-SSIM | LPIPS | DISTS | WD3 | Q-hat v0 |
|---|---|---|---|---|---|---|
| teacher | 24.974 | 0.8883 | 0.0783 | 0.0650 | 0.1952 | +0.272 |
| student | 25.042 | 0.8874 | 0.0917 | 0.0726 | 0.2331 | +0.115 |

Q-hat v0 student - teacher: mean -0.157, worst -0.809, images worse by > 0.1: 17/30

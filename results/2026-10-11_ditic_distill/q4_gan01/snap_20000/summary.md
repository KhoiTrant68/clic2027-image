# Student vs teacher, DiT-IC q4, 30 CLIC validation images

Checkpoint: distill/run_q4_gan01/snap_20000.pt (step 20000). Same bitstreams (mean 0.0996 bpp).

| decoder | PSNR | MS-SSIM | LPIPS | DISTS | WD3 | Q-hat v0 |
|---|---|---|---|---|---|---|
| teacher | 24.974 | 0.8883 | 0.0783 | 0.0650 | 0.1952 | +0.272 |
| student | 25.106 | 0.8876 | 0.0986 | 0.0784 | 0.2503 | -0.059 |

Q-hat v0 student - teacher: mean -0.331, worst -0.762, images worse by > 0.1: 30/30

# Student vs teacher, DiT-IC q3, 30 CLIC validation images

Checkpoint: distill/run_q3_gan01/snap_8000.pt (step 8000). Same bitstreams (mean 0.0762 bpp).

| decoder | PSNR | MS-SSIM | LPIPS | DISTS | WD3 | Q-hat v0 |
|---|---|---|---|---|---|---|
| teacher | 24.703 | 0.8828 | 0.0826 | 0.0692 | 0.2144 | +0.157 |
| student | 24.777 | 0.8805 | 0.1008 | 0.0813 | 0.2540 | -0.112 |

Q-hat v0 student - teacher: mean -0.269, worst -1.203, images worse by > 0.1: 29/30

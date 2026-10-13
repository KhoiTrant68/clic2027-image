# Student vs teacher, DiT-IC q2, 30 CLIC validation images

Checkpoint: distill/run_q2_gan01/snap_20000.pt (step 20000). Same bitstreams (mean 0.0400 bpp).

| decoder | PSNR | MS-SSIM | LPIPS | DISTS | WD3 | Q-hat v0 |
|---|---|---|---|---|---|---|
| teacher | 23.514 | 0.8434 | 0.1026 | 0.0914 | 0.2742 | -0.450 |
| student | 23.600 | 0.8439 | 0.1172 | 0.0990 | 0.3100 | -0.647 |

Q-hat v0 student - teacher: mean -0.197, worst -0.653, images worse by > 0.1: 24/30

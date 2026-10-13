# Student vs teacher, DiT-IC q2, 30 CLIC validation images

Checkpoint: distill/run_q2_gan01/snap_25000.pt (step 25000). Same bitstreams (mean 0.0400 bpp).

| decoder | PSNR | MS-SSIM | LPIPS | DISTS | WD3 | Q-hat v0 |
|---|---|---|---|---|---|---|
| teacher | 23.514 | 0.8434 | 0.1026 | 0.0914 | 0.2742 | -0.450 |
| student | 23.637 | 0.8446 | 0.1157 | 0.0975 | 0.3072 | -0.611 |

Q-hat v0 student - teacher: mean -0.161, worst -0.512, images worse by > 0.1: 20/30

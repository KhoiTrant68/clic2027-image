# Student vs teacher, DiT-IC q2, 30 CLIC validation images

Checkpoint: distill/run_q2_gan01/snap_30000.pt (step 30000). Same bitstreams (mean 0.0400 bpp).

| decoder | PSNR | MS-SSIM | LPIPS | DISTS | WD3 | Q-hat v0 |
|---|---|---|---|---|---|---|
| teacher | 23.514 | 0.8434 | 0.1026 | 0.0914 | 0.2742 | -0.450 |
| student | 23.645 | 0.8449 | 0.1141 | 0.0971 | 0.3082 | -0.584 |

Q-hat v0 student - teacher: mean -0.134, worst -0.572, images worse by > 0.1: 14/30

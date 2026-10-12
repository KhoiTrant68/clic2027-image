# Student vs teacher, DiT-IC q3, 30 CLIC validation images

Checkpoint: distill/run_q3_gan01/snap_14000.pt (step 14000). Same bitstreams (mean 0.0762 bpp).

| decoder | PSNR | MS-SSIM | LPIPS | DISTS | WD3 | Q-hat v0 |
|---|---|---|---|---|---|---|
| teacher | 24.703 | 0.8828 | 0.0826 | 0.0692 | 0.2144 | +0.157 |
| student | 24.614 | 0.8773 | 0.1073 | 0.0849 | 0.2619 | -0.208 |

Q-hat v0 student - teacher: mean -0.364, worst -1.280, images worse by > 0.1: 30/30

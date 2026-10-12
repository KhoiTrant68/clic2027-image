# Student vs teacher, DiT-IC q3, 30 CLIC validation images

Checkpoint: distill/run_q3_gan01/snap_10000.pt (step 10000). Same bitstreams (mean 0.0762 bpp).

| decoder | PSNR | MS-SSIM | LPIPS | DISTS | WD3 | Q-hat v0 |
|---|---|---|---|---|---|---|
| teacher | 24.703 | 0.8828 | 0.0826 | 0.0692 | 0.2144 | +0.157 |
| student | 24.852 | 0.8812 | 0.1007 | 0.0798 | 0.2542 | -0.086 |

Q-hat v0 student - teacher: mean -0.243, worst -1.220, images worse by > 0.1: 27/30

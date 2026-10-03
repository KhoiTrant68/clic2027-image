# Go/no-go 1 (a): exact noise model vs Gaussian assumption

Train latents: 792; threshold: exact must beat gaussian by ≥ 5% latent MSE at ≥ 2 of the rates (per test set).

Final train loss (mean of last 500 steps): exact 0.1350, gaussian 0.1353, nodither 0.1562

## train_holdout (8 latents)

| target bpp | Δ | raw dither | exact | gaussian | exact vs gaussian | raw no-dither | no-dither denoiser |
|---|---|---|---|---|---|---|---|
| 0.010 | 3.012 | 0.7564 | 0.2289 | 0.2293 | +0.2% | 0.3841 | 0.2911 |
| 0.020 | 2.136 | 0.3803 | 0.1616 | 0.1620 | +0.2% | 0.2661 | 0.1891 |
| 0.030 | 1.629 | 0.2212 | 0.1176 | 0.1178 | +0.2% | 0.1819 | 0.1289 |
| 0.050 | 0.984 | 0.0808 | 0.0583 | 0.0584 | +0.2% | 0.0767 | 0.0600 |

**train_holdout: 0/4 valid rates reach the threshold -> FAIL**

## kodak (24 latents)

| target bpp | Δ | raw dither | exact | gaussian | exact vs gaussian | raw no-dither | no-dither denoiser |
|---|---|---|---|---|---|---|---|
| 0.010 | 3.012 | 0.7555 | 0.2284 | 0.2288 | +0.2% | 0.3919 | 0.2937 |
| 0.020 | 2.136 | 0.3801 | 0.1630 | 0.1633 | +0.2% | 0.2779 | 0.1926 |
| 0.030 | 1.629 | 0.2211 | 0.1190 | 0.1193 | +0.2% | 0.1912 | 0.1308 |
| 0.050 | 0.984 | 0.0807 | 0.0596 | 0.0598 | +0.2% | 0.0791 | 0.0604 |

**kodak: 0/4 valid rates reach the threshold -> FAIL**

## clic2020_valid (41 latents)

| target bpp | Δ | raw dither | exact | gaussian | exact vs gaussian | raw no-dither | no-dither denoiser |
|---|---|---|---|---|---|---|---|
| 0.010 | 3.012 | 0.7566 | 0.1980 | 0.1982 | +0.1% | 0.3350 | 0.2556 |
| 0.020 | 2.136 | 0.3804 | 0.1444 | 0.1446 | +0.1% | 0.2456 | 0.1756 |
| 0.030 | 1.629 | 0.2211 | 0.1078 | 0.1080 | +0.2% | 0.1754 | 0.1228 |
| 0.050 | 0.984 | 0.0807 | 0.0559 | 0.0560 | +0.2% | 0.0771 | 0.0579 |

**clic2020_valid: 0/4 valid rates reach the threshold -> FAIL**

Note: the no-dither arm uses the same Δ, so its true rate is lower than the dithered arms'; compare it only together with the rate table in results.json.
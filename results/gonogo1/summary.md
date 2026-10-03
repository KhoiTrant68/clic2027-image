# Go/no-go 1 (a): exact noise model vs Gaussian assumption

Train latents: 39; threshold: exact must beat gaussian by ≥ 5% latent MSE at ≥ 2 of the rates (per test set).

## kodak (24 latents)

| target bpp | Δ | raw dither | exact | gaussian | exact vs gaussian | raw no-dither | no-dither denoiser |
|---|---|---|---|---|---|---|---|
| 0.010 | 1.150 | 0.1101 | 0.2007 | 0.2313 | +13.2% | 0.1065 | 0.1866 |
| 0.020 | 0.801 | 0.0535 | 0.1645 | 0.2024 | +18.7% | 0.0532 | 0.1384 |
| 0.030 | 0.609 | 0.0309 | 0.1394 | 0.1823 | +23.5% | 0.0308 | 0.1096 |
| 0.050 | 0.370 | 0.0114 | 0.0989 | 0.1473 | +32.8% | 0.0114 | 0.0684 |

**kodak: 4/4 rates reach the threshold -> PASS**

## clic2020_valid (41 latents)

| target bpp | Δ | raw dither | exact | gaussian | exact vs gaussian | raw no-dither | no-dither denoiser |
|---|---|---|---|---|---|---|---|
| 0.010 | 1.150 | 0.1102 | 0.1574 | 0.1686 | +6.7% | 0.1011 | 0.1488 |
| 0.020 | 0.801 | 0.0535 | 0.1338 | 0.1478 | +9.5% | 0.0524 | 0.1121 |
| 0.030 | 0.609 | 0.0309 | 0.1169 | 0.1336 | +12.5% | 0.0307 | 0.0898 |
| 0.050 | 0.370 | 0.0114 | 0.0885 | 0.1092 | +19.0% | 0.0114 | 0.0586 |

**clic2020_valid: 4/4 rates reach the threshold -> PASS**

Note: the no-dither arm uses the same Δ, so its true rate is lower than the dithered arms'; compare it only together with the rate table in results.json.
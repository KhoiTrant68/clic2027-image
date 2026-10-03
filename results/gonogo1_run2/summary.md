# Go/no-go 1 (a): exact noise model vs Gaussian assumption

Train latents: 39; threshold: exact must beat gaussian by ≥ 5% latent MSE at ≥ 2 of the rates (per test set).

## kodak (24 latents)

| target bpp | Δ | raw dither | exact | gaussian | exact vs gaussian | raw no-dither | no-dither denoiser |
|---|---|---|---|---|---|---|---|
| 0.010 | 1.150 | 0.1101 | 0.1981 | 0.2277 | +13.0% | 0.1065 | 0.1865 |
| 0.020 | 0.801 | 0.0535 | 0.1640 | 0.1982 | +17.3% | 0.0532 | 0.1390 |
| 0.030 | 0.609 | 0.0309 | 0.1415 | 0.1770 | +20.1% | 0.0308 | 0.1099 |
| 0.050 | 0.370 | 0.0114 | 0.1058 | 0.1396 | +24.2% | 0.0114 | 0.0689 |

**kodak: 4/4 rates reach the threshold -> PASS**

## clic2020_valid (41 latents)

| target bpp | Δ | raw dither | exact | gaussian | exact vs gaussian | raw no-dither | no-dither denoiser |
|---|---|---|---|---|---|---|---|
| 0.010 | 1.150 | 0.1102 | 0.1444 | 0.1558 | +7.4% | 0.1011 | 0.1507 |
| 0.020 | 0.801 | 0.0535 | 0.1204 | 0.1337 | +9.9% | 0.0524 | 0.1141 |
| 0.030 | 0.609 | 0.0309 | 0.1047 | 0.1186 | +11.7% | 0.0307 | 0.0917 |
| 0.050 | 0.370 | 0.0114 | 0.0801 | 0.0923 | +13.3% | 0.0114 | 0.0602 |

**clic2020_valid: 4/4 rates reach the threshold -> PASS**

Note: the no-dither arm uses the same Δ, so its true rate is lower than the dithered arms'; compare it only together with the rate table in results.json.
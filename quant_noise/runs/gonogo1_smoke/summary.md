# Go/no-go 1 (a): exact noise model vs Gaussian assumption

Train latents: 24; threshold: exact must beat gaussian by ≥ 5% latent MSE at ≥ 2 of the rates (per test set).

## synthetic_test (6 latents)

| target bpp | Δ | raw dither | exact | gaussian | exact vs gaussian | raw no-dither | no-dither denoiser |
|---|---|---|---|---|---|---|---|
| 0.010 | 5.263 | 2.3128 | 1.0990 | 1.1112 | +1.1% | 1.3306 | 1.0183 |
| 0.020 | 3.271 | 0.8930 | 0.6970 | 0.7007 | +0.5% | 0.7392 | 0.5866 |
| 0.030 | 2.297 | 0.4388 | 0.5176 | 0.5206 | +0.6% | 0.4187 | 0.3495 |
| 0.050 | 1.272 | 0.1350 | 0.2678 | 0.2811 | +4.8% | 0.1347 | 0.1254 |

**synthetic_test: 0/4 rates reach the threshold -> FAIL**

Note: the no-dither arm uses the same Δ, so its true rate is lower than the dithered arms'; compare it only together with the rate table in results.json.
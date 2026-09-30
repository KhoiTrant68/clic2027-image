# Quantization error vs. Gaussian — 8 latents

Medians over coefficients / latent channels. nd = no dither, d = subtractive dither, g = Gaussian reference.

## direct

| target bpp | Δ | zero frac (nd) | kurt coef nd / d / g | kurt latent nd / d / g | corr(e,y) coef nd / d | corr(e,y) latent nd / d | offdiag latent nd / d / g | lag-1 latent nd / d / g | content ρ nd / d / g | MSE latent nd / d |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.010 | 6.57 | 1.00 | +0.59 / -1.20 / -0.00 | +0.59 / -1.20 / -0.00 | -0.92 / -0.00 | -0.92 / -0.00 | +0.15 / +0.00 / +0.00 | +0.66 / +0.00 / +0.00 | +0.64 / -0.07 / -0.01 | 2.736 / 10.078 |
| 0.020 | 5.88 | 0.99 | +0.39 / -1.20 / -0.00 | +0.39 / -1.20 / -0.00 | -0.85 / -0.00 | -0.85 / -0.00 | +0.13 / +0.00 / +0.00 | +0.60 / -0.00 / -0.00 | +0.64 / +0.03 / -0.01 | 2.674 / 8.041 |
| 0.030 | 5.42 | 0.99 | +0.22 / -1.20 / +0.00 | +0.22 / -1.20 / +0.00 | -0.79 / +0.00 | -0.79 / +0.00 | +0.12 / +0.00 / +0.00 | +0.56 / +0.00 / -0.00 | +0.63 / +0.02 / +0.03 | 2.606 / 6.834 |
| 0.050 | 4.81 | 0.97 | -0.03 / -1.20 / +0.00 | -0.03 / -1.20 / +0.00 | -0.68 / +0.00 | -0.68 / +0.00 | +0.09 / +0.00 / +0.00 | +0.48 / +0.00 / -0.00 | +0.62 / -0.03 / -0.07 | 2.468 / 5.389 |

## klt2

| target bpp | Δ | zero frac (nd) | kurt coef nd / d / g | kurt latent nd / d / g | corr(e,y) coef nd / d | corr(e,y) latent nd / d | offdiag latent nd / d / g | lag-1 latent nd / d / g | content ρ nd / d / g | MSE latent nd / d |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.010 | 15.50 | 0.99 | +0.05 / -1.20 / +0.00 | +0.38 / -0.06 / -0.00 | -1.00 / +0.00 | -0.72 / -0.00 | +0.16 / +0.00 / +0.00 | +0.59 / +0.00 / -0.00 | +0.54 / +0.06 / -0.02 | 2.400 / 20.042 |
| 0.020 | 11.97 | 0.98 | +0.03 / -1.20 / +0.00 | +0.19 / -0.07 / +0.01 | -1.00 / +0.00 | -0.56 / +0.00 | +0.15 / +0.00 / +0.00 | +0.51 / -0.00 / -0.00 | +0.46 / -0.01 / +0.02 | 2.060 / 11.964 |
| 0.030 | 9.84 | 0.97 | +0.02 / -1.20 / -0.02 | +0.10 / -0.06 / +0.00 | -1.00 / -0.00 | -0.47 / +0.00 | +0.14 / +0.00 / +0.00 | +0.45 / -0.00 / -0.00 | +0.42 / +0.05 / +0.01 | 1.774 / 8.064 |
| 0.050 | 7.36 | 0.95 | -0.01 / -1.20 / -0.01 | +0.04 / -0.06 / +0.01 | -1.00 / -0.00 | -0.37 / +0.00 | +0.13 / +0.00 / +0.00 | +0.36 / -0.00 / +0.00 | +0.40 / +0.02 / +0.03 | 1.372 / 4.517 |

## Go/no-go 1 (proposed thresholds)

Thresholds: {'abs_kurt': 0.5, 'abs_corr_ey': 0.3, 'offdiag': 0.1, 'content_rho': 0.3}

- direct @ 0.010 bpp: NON-GAUSSIAN (coef_kurt, latent_kurt, coef_corr_ey, latent_corr_ey, latent_offdiag, latent_content)
- direct @ 0.020 bpp: NON-GAUSSIAN (coef_corr_ey, latent_corr_ey, latent_offdiag, latent_content)
- direct @ 0.030 bpp: NON-GAUSSIAN (coef_corr_ey, latent_corr_ey, latent_offdiag, latent_content)
- direct @ 0.050 bpp: NON-GAUSSIAN (coef_corr_ey, latent_corr_ey, latent_content)
- klt2 @ 0.010 bpp: NON-GAUSSIAN (coef_corr_ey, latent_corr_ey, latent_offdiag, latent_content)
- klt2 @ 0.020 bpp: NON-GAUSSIAN (coef_corr_ey, latent_corr_ey, latent_offdiag, latent_content)
- klt2 @ 0.030 bpp: NON-GAUSSIAN (coef_corr_ey, latent_corr_ey, latent_offdiag, latent_content)
- klt2 @ 0.050 bpp: NON-GAUSSIAN (coef_corr_ey, latent_corr_ey, latent_offdiag, latent_content)

**All tested low rates non-Gaussian (no dither): True**
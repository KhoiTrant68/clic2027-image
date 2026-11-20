# Pha 0: entropy model trên latent MLIC++ cố định (CLIC30, bpp ước lượng)

## q1 (MLIC++ R = 0.1175 bpp)

| Biến thể | Mô tả | bpp | so với R | so với V0 | lấy lại khoảng cách | forward ×V0 | tham số |
|---|---|---:|---:|---:|---:|---:|---:|
| V0 | GC hiện tại (số nguyên) | 0.1693 | +44.10% | +0.00% | 0% | 1.00 | 4.3 M |
| V0f | GC hiện tại, float | 0.1672 | +42.27% | -1.27% | 4% | 0.50 | 4.3 M |
| V1 | + E1 5×5 cục bộ + E2 mạng sâu | 0.1621 | +37.96% | -4.26% | 14% | 1.16 | 7.2 M |
| V2 | + E3 toàn cục giữa slice | 0.1616 | +37.51% | -4.58% | 15% | 2.00 | 7.9 M |

(50k bước, chỉ để chọn biến thể) R thấp hơn V0 30.61%; tiêu chí G0 áp ở 50k: entropy là nguồn lợi thế; biến thể chọn: không có biến thể đạt ≥ 70% với forward ≤ 2× V0

## q2 (MLIC++ R = 0.1798 bpp)

| Biến thể | Mô tả | bpp | so với R | so với V0 | lấy lại khoảng cách | forward ×V0 | tham số |
|---|---|---:|---:|---:|---:|---:|---:|
| V0 | GC hiện tại (số nguyên) | 0.2481 | +38.01% | +0.00% | 0% | 1.00 | 4.3 M |
| V0f | GC hiện tại, float | 0.2474 | +37.60% | -0.30% | 1% | 0.38 | 4.3 M |
| V1 | + E1 5×5 cục bộ + E2 mạng sâu | 0.2412 | +34.17% | -2.78% | 10% | 1.38 | 7.2 M |
| V2 | + E3 toàn cục giữa slice | 0.2405 | +33.78% | -3.07% | 11% | 1.28 | 7.9 M |

(50k bước, chỉ để chọn biến thể) R thấp hơn V0 27.54%; tiêu chí G0 áp ở 50k: entropy là nguồn lợi thế; biến thể chọn: không có biến thể đạt ≥ 70% với forward ≤ 2× V0


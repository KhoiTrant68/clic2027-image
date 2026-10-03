# Parity: ratflow.nn / ratflow.entropy so với diffusers và giữa các thiết bị

| kiểm tra | giá trị | ngưỡng | kết quả |
|---|---|---|---|
| DC-AE encode khớp diffusers (`dcae/encode`) | 0 | 1e-05 | PASS |
| DC-AE decode khớp diffusers (`dcae/decode`) | 0 | 1e-05 | PASS |
| DC-AE encode theo tile khớp (`dcae/tiled_encode`) | 0 | 1e-05 | PASS |
| DC-AE decode theo tile khớp (`dcae/tiled_decode`) | 0 | 1e-05 | PASS |
| SANA 0.6B fp32 khớp (`dit/0.6B_float32`) | 0 | 0.0001 | PASS |
| SANA 1.6B fp16 khớp (`dit/1.6B_float16`) | 0 | 0.005 | PASS |
| h_s số nguyên: CPU == GPU từng bit (`entropy/hs_int_cpu_vs_gpu`) | True | True | PASS |
| đường float (train) sát đường số nguyên (`entropy/hs_float_vs_int`) | 0 | 0.001 | PASS |
| nén/giải nén khớp hoàn toàn (`entropy/roundtrip`) | True | True | PASS |

`dcae/decode_2048x1360_fp32_s`: 6.7401754061381025

`entropy/acc_bound_log2`: 35.25574481936111

`entropy/roundtrip`: {'ok': True, 'bytes_y': 426534, 'bytes_z': 34600, 'ideal_bytes_y': 425664.25, 'n_y': 1966080}
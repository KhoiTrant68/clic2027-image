# Parity: ratflow.nn / ratflow.entropy so với diffusers và giữa các thiết bị

| kiểm tra | giá trị | ngưỡng | kết quả |
|---|---|---|---|
| DC-AE encode khớp diffusers (`dcae/encode`) | 0 | 1e-05 | PASS |
| DC-AE decode khớp diffusers (`dcae/decode`) | 0 | 1e-05 | PASS |
| DC-AE encode theo tile khớp (`dcae/tiled_encode`) | 0 | 1e-05 | PASS |
| DC-AE decode theo tile khớp (`dcae/tiled_decode`) | 0 | 1e-05 | PASS |
| SANA 0.6B fp32 khớp (`dit/0.6B_float32`) | 0 | 0.0001 | PASS |
| SANA 1.6B fp16 khớp (`dit/1.6B_float16`) | – | – | CHƯA CHẠY |
| h_s số nguyên: CPU == GPU từng bit (`entropy/hs_int_cpu_vs_gpu`) | – | – | CHƯA CHẠY |
| đường float (train) sát đường số nguyên (`entropy/hs_float_vs_int`) | – | – | CHƯA CHẠY |
| nén/giải nén khớp hoàn toàn (`entropy/roundtrip`) | – | – | CHƯA CHẠY |

`dcae/decode_2048x1360_fp32_s`: 6.88040033976237

`entropy/acc_bound_log2`: 35.25574481936111
# A0: kiểm tra độ tất định của entropy model

**Câu hỏi cần trả lời:** cùng một `h_s` và cùng một `ẑ`, chạy trên các thiết bị khác nhau thì `scale_idx` có giống hệt nhau không? Chỉ cần lệch một phần tử là bitstream giải mã trên server bị hỏng.

`probe.py` thử 4 chế độ:

| Chế độ | Cách tính |
|---|---|
| `float32` | fp32, tắt TF32 |
| `float32tf` | fp32, bật TF32 (mặc định cho conv trên L4) |
| `bf16` | bf16 |
| `intsim` | Mạng số nguyên (weight int8), tính bằng float64 nhưng mọi giá trị đều là số nguyên < 2^53, nên phép tính chính xác và không phụ thuộc thứ tự cộng |

Input và weights được sinh bằng numpy, nên giống hệt nhau trên mọi máy.

## Chạy

**Trên L4 thuê** (trong image `clic-gpu`, giống server nhất):
```bash
sudo docker run --rm --gpus all -v $PWD:/w -w /w clic-gpu python3 probe.py --device cuda --tag l4
sudo docker run --rm -v $PWD:/w -w /w clic-gpu python3 probe.py --device cpu --tag l4cpu
```

**Trên Kaggle** (T4, rồi đổi sang P100): `pip install -q torch==2.6.0`, sau đó chạy `python probe.py --device cuda --tag t4`.

**So sánh:** gom các file `out/*.npz` về một chỗ, rồi chạy:
```bash
python compare.py out/*.npz
```

## Kết quả mong đợi

- `intsim`: **OK** trên mọi cặp thiết bị. Đây là cơ sở cho A2.
- Các chế độ float: có mismatch giữa các GPU khác nhau, có thể cả L4 so với CPU.
- Nếu float **khớp** giữa hai lần chạy trên L4 (hai máy thuê khác nhau, hoặc khác driver), thì có thể dùng tạm cách "encode trên L4 trong image `clic-gpu`" cho các lần nộp sớm, trong lúc chờ A2 xong.

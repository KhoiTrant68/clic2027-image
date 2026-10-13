# Kit đo DiT-IC trên L4 thuê

Kit trả lời hai câu hỏi trước mốc quyết định 27/10:

1. **Tốc độ:** trên L4, decoder DiT-IC q3 giải 30 ảnh validation mất bao lâu, với decoder DC-AE gốc (thầy) và bản chưng cất (trò)?
   Cách đo giống CLIC: mỗi lần chạy là một tiến trình mới, mỗi ảnh giải đúng một lần.
2. **Tất định (hướng A):** bitstream nén trên một L4 có được giải đúng từng bit trên một L4 *khác* không?
   Xem `results/2026-10-12_ditic_determinism/summary.md`.

Môi trường dựng theo `devkit/docker/requirements.txt` của CLIC: torch 2.6.0 cu124, cuDNN 9.1.0.70, numpy 1.26.4.
Thêm diffusers, compressai, timm cho DiT-IC. Codec chạy fp32 với cuDNN deterministic và tắt TF32.
DiT và DC-AE chạy bf16, giải cả khung hình, latent gom bội số 16.

## Thuê máy

- **GPU:** 1 × NVIDIA L4 24 GB.
- **Hệ điều hành:** Ubuntu 22.04 hoặc 24.04, driver NVIDIA ≥ 550. Container (RunPod, vast.ai) hay VM đều được.
- **Đĩa:** ≥ 40 GB trống. Cần mạng tới PyPI, GitHub, HuggingFace.
- **Phần mềm:** chỉ cần `curl` và `git`. `uv` tự cài Python 3.12.
- **Thời gian:** khoảng 30 phút mỗi lần: dựng môi trường 10–15 phút (tải torch ~2.5 GB, checkpoint q3 4.3 GB), đo ~10 phút.
- **Thuê 2 lần, trên 2 máy vật lý khác nhau.** Ví dụ: thuê, chạy, trả máy rồi thuê lại, hoặc dùng hai nhà cung cấp.
  Lần 2 cần file bitstream của lần 1.

## Chạy

Kit đóng gói sẵn trên server: `guess:clic/clic27_l4_kit.tgz` (~28 MB).
Chứa thư mục này, `ditic_distill/common.py`, decoder trò `students/q3.pt` và `streams_l40.pt` (30 ảnh nén trên L40).

**Lần 1 (máy L4 thứ nhất):**

```bash
scp -3 guess:clic/clic27_l4_kit.tgz L4:      # L4 = tên máy thuê trong ~/.ssh/config
ssh L4
tar xzf clic27_l4_kit.tgz
nohup bash clic27_l4_kit/l4_kit/run_all.sh > run.log 2>&1 &
tail -f run.log                               # chờ dòng "results: ..."
```

Lấy về `~/clic27_l4/l4_results_<host>.tgz`. File này có `out_<host>/streams_<host>.pt`: 30 ảnh nén trên L4 thứ nhất.

**Lần 2 (máy L4 khác):** làm như lần 1, nhưng trước khi chạy, chép `streams_<host lần 1>.pt` vào `clic27_l4_kit/`.
`run_all.sh` giải mọi file `streams_*.pt` có trong kit, hoặc truyền đường dẫn làm tham số.

Nếu thêm máy thuê vào `~/.ssh/config` của máy này, mình có thể tự chạy và đọc kết quả.

## Đọc kết quả

Các dòng bắt đầu bằng `==` trong `run.log`:

- `== BIT-EXACT: A -> B`: giải bitstream nén trên A, tại B, không lệch chỉ số scale hay ký hiệu nào.
  `MISMATCH on k/30` nghĩa là hướng A không đủ.
- `== teacher|student on NVIDIA L4: load … | first image … | images 2..30 mean … | decode total …`.
  `load` là dựng mạng fp32 từ checkpoint 4.3 GB rồi đổi sang bf16. Bản nộp sẽ gói sẵn trọng số bf16 nên nạp nhanh hơn nhiều.

Lần 1 còn giải `streams_l40.pt`, tức kiểm tra L40 → L4. Hai GPU khác loại nên rất có thể lệch; con số này chỉ để biết.

So sánh offline hai file streams, không cần GPU:

```bash
python l4_kit/l4_check.py compare streams_A.pt streams_B.pt
```

Nếu hai bitstream giống hệt và mỗi máy tự kiểm BIT-EXACT, thì máy này giải đúng bitstream của máy kia.

## Số đối chiếu trên L40 của server (13/10, cùng kit, cùng lệnh)

| decoder | nạp | ảnh đầu | ảnh 2..30 (trung bình) | giải 30 ảnh |
|---|---|---|---|---|
| thầy | 17–19 s | 0.8–0.9 s | 0.57 s | 17.3–17.5 s |
| trò | 19–21 s | 0.6–0.7 s | 0.37 s | 11.4 s |

- Tự kiểm L40 → L40: BIT-EXACT.
- Peak VRAM: thầy 5.5 GB, trò 4.4 GB.
- Mốc tham chiếu: đội nhất GPU 2025 (Vcoder) giải 30 ảnh trong 24 s ở 0.075 bpp.

## Đóng gói lại kit (trên server)

```bash
bash make_kit.sh ~/clic/l4kit ~/clic/distill/students ~/clic/clic27_l4/out_server-01/streams_server-01.pt ~/clic/clic27_l4_kit.tgz
```

`~/clic/l4kit` chứa bản sao `experiments/l4_kit` và `experiments/ditic_distill/common.py` của repo.

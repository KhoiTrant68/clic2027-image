# Kiểm gói Mixed ở 0,075 bpp và đo thời gian giải mã

Gói decoder dùng mã MLIC++ tại commit 4aa9d2a512eb9db382172058e6c278fb0d376e17, trọng số q2 legacy đã bỏ trạng thái optimizer, và checkpoint Mixed. Gói dữ liệu dùng các stream 0,075 bpp đã đóng băng. Đổi trường decoder trong manifest từ Perceptual sang Mixed, giữ nguyên byte của từng stream và tổng kích thước 802.034 byte.

Các gói, ảnh đầu ra và log nằm trên server tại /home/jovyan/clic2027-artifacts/mlicpp/mixed-clic-validation-0075-20261112. Bộ dựng gói lưu một danh sách hash pixel độc lập từ kết quả Mixed trước đó. [Mã dựng gói](../experiments/mlicpp/build_mixed_validation.sh) · [Entry point và bộ đo](../experiments/mlicpp/clic_mixed_decode.py)

## Kiểm khả năng tái lập

Lượt thử trên CPU xuất đúng kích thước 1.360 × 2.048 nhưng sai ảnh nghiêm trọng so với bản GPU. MAE là 142,57 trên thang 0–255 và mọi pixel có ít nhất một kênh khác. Một ảnh thử trên L40 trong venv MLIC++ khớp SHA-256 pixel với bản Mixed đã lưu. Tương thích CPU/GPU chưa được xác lập.

Docker CLI có mặt nhưng không kết nối được daemon trên server. Phép thử gọi entry point giải nén từ decoder.zip trong venv hiện có. Container devkit và GPU L4 còn cần kiểm riêng.

Lượt đầy đủ giải mã từ ZIP trên L40 xuất đúng 30 PNG và khớp cả 30 hash pixel. Trần 4 GiB gặp OOM tại ảnh 16 trong synthesis. Lượt hoàn tất dùng trần 5 GiB và nhả cache của decoder giữa các ảnh. Thời gian toàn tiến trình là 100,027 giây, gồm 46,556 giây bên trong model, trung vị 1,542 giây mỗi ảnh. decoder.zip có 537.822.278 byte và SHA-256 ff2f5c8131ddb45823f0b4bdb7a6f5fcbce5c9aa29389b5091acd3a9ee039f7e. bs.zip có 802.034 byte, tương đương 0,074396 bpp, và SHA-256 3af2e1c58e251e97598890c4c534e03bfe3b42f94a6a190fa8d541155981dd9e.

### Bản vá kiểm tra thiết bị và môi trường giải mã

Decoder đọc trường `entropy_device` trước khi nạp model. Gói Mixed hiện có mặc định dùng CUDA và trả lỗi ngay khi được gọi với `--device cpu`. Phép thử trên server xác nhận lệnh trả lỗi và thư mục ảnh đầu ra chưa được tạo.

Bản vá được đóng gói riêng thành `decoder-guarded.zip` cùng thư mục với bản gốc trên server. Kiểm tra ZIP xác nhận các mục ngoài mã decoder giữ nguyên kích thước, CRC và quyền tệp. Bản mới có 537.822.527 byte và SHA-256 `f82bf8c7e54ddb4eeb24612e3f3c4d6fa67417d8d94ed8c4bebde945c0f16927`. Mốc 30/30 ảnh trên L40 thuộc bản gốc.

Một ảnh được mã hóa lại bằng CPU với cùng trọng số và hệ số 4,0. Luồng hyperprior trùng với bản CUDA, còn luồng latent chính khác. Payload CPU và CUDA lần lượt là 35.023 và 35.027 byte. Tiến trình CPU độc lập giải mã đúng hash pixel và kích thước 1.360 × 2.048 trong môi trường ngắt mạng. Thời gian giải mã ảnh này là 32,878 giây. Phép thử xác nhận cặp mã hóa và giải mã CPU trên một ảnh.

Dockerfile khóa các phiên bản của môi trường đã chạy đúng trên L40, gồm PyTorch 2.0.1, CUDA 11.8 và CompressAI 1.2.0b3. Môi trường mặc định của devkit dùng PyTorch 2.6 và CUDA 12.4. Devkit có hướng dẫn đăng ký môi trường riêng. [Hướng dẫn của CLIC](https://github.com/clic-challenge/devkit#requesting-a-different-environment)

Trên máy Linux có Docker và NVIDIA L4, chạy từ thư mục gốc của repo với ba tệp đã sao chép từ server. Script chạy decoder với 2 CPU, 12 GiB RAM, trần bộ nhớ GPU 5 GiB và tắt mạng. Kết quả cần đạt là `Verified 30/30 PNGs`.

```bash
docker build --platform linux/amd64 -f clic/l4/Dockerfile.mlicpp -t clic-mlicpp:cu118 clic/l4
bash clic/l4/verify_mixed_container.sh decoder-guarded.zip bs.zip expected.json verify-l4
```

Server hiện tại không truy cập được Docker daemon. Docker image và lần đối chiếu đầy đủ trên L4 vẫn chưa được kiểm chứng.

### Chạy trực tiếp trên server

Lượt 30/30 ảnh trên L40 đã chạy trực tiếp bằng môi trường Python 3.10.22 ở `/home/jovyan/clic2027-envs/mlicpp`. Miniconda cũng có thể dùng để tạo môi trường riêng nếu cần chuyển máy. Khi tái lập, giữ các phiên bản PyTorch, CUDA và CompressAI đã ghi ở trên.

Trên server, thư mục sau đã chứa decoder đã vá, `bs.zip` và `expected.json`. Khi có L40 đủ bộ nhớ trống, thay `0` bằng chỉ số GPU đã chọn. `unshare -Urn` ngắt mạng trong lượt chạy.

```bash
cd /home/jovyan/clic2027-artifacts/mlicpp/mixed-clic-validation-0075-20261112/native-guarded.EVCP2w
export MLIC_PYTHON=/home/jovyan/clic2027-envs/mlicpp/bin/python
CUDA_VISIBLE_DEVICES=0 unshare -Urn ./decode --memory-mib 5120 --report decoded.json
"$MLIC_PYTHON" - <<'PY'
import hashlib
import json
from pathlib import Path

from PIL import Image

expected = json.loads(Path("expected.json").read_text())
images = {path.name: path for path in Path("images").glob("*.png")}
if set(images) != set(expected):
    raise SystemExit("Image inventory mismatch")
for name, digest in expected.items():
    with Image.open(images[name]) as image:
        if hashlib.sha256(image.convert("RGB").tobytes()).hexdigest() != digest:
            raise SystemExit(f"Pixel hash mismatch: {name}")
print(f"Verified {len(images)}/{len(expected)} PNGs")
PY
```

## Phạm vi bộ đo

Lượt chạy thông thường đo cả thời gian toàn tiến trình và thời gian gọi model trên từng ảnh. Lượt chẩn đoán đồng bộ CUDA ở ranh giới các phần để đo mô hình ngữ cảnh, LRP, hyperprior, lập chỉ số CDF và synthesis. Thời gian rANS chỉ gồm lệnh đọc symbol của decoder chính. Các phép chuyển tensor sang danh sách, ghép tensor và chi phí gọi Python nằm trong thời gian chưa phân loại. GPU 1 có tải đồng thời, vì vậy các số đo này mô tả lượt chạy trên server và cần đo lại trên phần cứng dành riêng để so hiệu năng.

Lượt chẩn đoán cũng khớp 30/30 hash pixel, với 58,015 giây bên trong model.

| Phần | Giây | Tỷ lệ |
| --- | ---: | ---: |
| Chưa quy vào một phần cụ thể | 22,070 | 38,04% |
| Mô hình ngữ cảnh và dự đoán tham số entropy | 12,752 | 21,98% |
| Synthesis | 11,585 | 19,97% |
| rANS đọc symbol chính | 4,624 | 7,97% |
| LRP | 4,290 | 7,39% |
| Lập chỉ số CDF | 1,706 | 2,94% |
| Giải mã và tái tạo hyperprior | 0,989 | 1,70% |

Ngữ cảnh và LRP chiếm gần 29,4% thời gian trong lượt đo. Phần chưa quy cụ thể còn lớn, nên cần tách các lần đồng bộ và chuyển dữ liệu CPU/GPU trước khi ước lượng lợi ích của thay lịch giải mã.

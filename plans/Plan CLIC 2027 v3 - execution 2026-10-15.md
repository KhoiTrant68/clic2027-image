# Kế hoạch CLIC 2027 v3 — từ repo hiện tại tới bài có khả năng tranh hạng nhất

Ngày lập: 15/10/2026. Phạm vi: Image GPU, đủ 0.075 / 0.15 / 0.3 bpp. Thay thế lịch và ưu tiên của v2; giữ v2 và bản gốc làm lịch sử.

Đây là kế hoạch đề xuất, chưa phải các job đã khởi chạy. Không có cách bảo đảm hạng nhất khi chưa biết đối thủ và bộ test. Mục tiêu là tối đa hóa cơ hội thắng bằng chất lượng người chấm, tốc độ thực tế và độ tin cậy của bài nộp.

**Cập nhật sau yêu cầu thử model mới:** đã hoàn tất vòng inference 204 điểm, gồm PULSE-L trên đủ 30 ảnh. Mix thêm PULSE-L đạt Q̂ 0.1841 / 0.7891 / 1.4912, so baseline 0.1836 / 0.7582 / 1.4215; chưa có chấm mù cho mix mới. Đưa PULSE-L, allocator có chi phí thời gian/load và fine-tune xuống rate thấp vào danh sách ưu tiên trước khi cam kết train dài. Kết quả, nguồn nghiên cứu, kiểm chứng và ablation đề xuất nằm trong `reports/CLIC2027_new_models_2026-10-15.md`. Lịch/ngân sách dưới đây là đề xuất ban đầu, cần áp dụng thứ tự cập nhật này. Người dùng đã nâng giới hạn server lên 30 GB/GPU.

## 1. Quyết định chiến lược

1. Tạo một bài nộp hợp lệ ngay trong tháng 10–11; duy trì nó trong suốt quá trình nghiên cứu.
2. Dùng server `ssh guess`, 2 × L40, làm compute chính. Người dùng đã nâng giới hạn từ nửa VRAM lên **30 GB trên mỗi GPU** trong cuộc trao đổi này. Áp dụng bảo thủ 30,000,000,000 byte cho tổng bộ nhớ thiết bị, chừa headroom cho dịch vụ/context; một job/GPU. Các pilot ngày 15/10 vẫn giữ guard cũ chặt hơn (nửa thiết bị, allocator tối đa 18 GiB) vì không cần tăng. L4 dùng để đo và kiểm chứng triển khai; không chờ thuê L4 mới bắt đầu nghiên cứu trên L40.
3. Giữ DiT-IC làm ứng viên chính ở 0.075; MS-ILLM và phối hợp theo ảnh làm baseline ở 0.15/0.3; giữ VTM-SCC cho nội dung chữ khi kiểm tra thực tế chứng minh cần nó.
4. Dồn phần lớn ngân sách cải thiện chất lượng vào 0.15. Song song hai hướng có giới hạn: mở rộng DiT-IC lên bitrate cao và cải thiện MS-ILLM.
5. Tối ưu encoder, chia bit, chọn seed/mode trước các thay đổi decoder lớn nếu lợi ích tương đương. Mọi metadata phải nằm trong bitstream và được tính byte.
6. Chốt bằng chấm mù và tập ảnh độc lập. Q̂ v0 là công cụ sàng lọc; không dùng nó làm bằng chứng duy nhất về khả năng thắng.

## 2. Xuất phát điểm đã được xác nhận

Nguồn: báo cáo 14/10, kết quả local và kiểm tra trực tiếp server trong cuộc trao đổi ngày 15/10. Không có lần train mới được thực hiện để viết kế hoạch này.

| Hạng mục | Hiện trạng | Hệ quả |
|---|---|---|
| Chất lượng mix, Q̂ v0 | +0.184 / +0.758 / +1.421 | Baseline cố định cho ba rate |
| DiT-IC riêng ở 0.075 | +0.164; chấm mù thắng MS-ILLM 62/90, một người chấm | Có tín hiệu tốt, cần thêm người và ảnh |
| DiT-IC ở bitrate cao | Các checkpoint hiện có chỉ tới khoảng 0.10 bpp | Chưa có bằng chứng chỉ tăng bitrate sẽ thắng MS-ILLM |
| Chưng cất | Student giảm 0.134 / 0.168 / 0.157 Q̂ ở q2/q3/q4 | Giữ làm lựa chọn tốc độ, không mặc định thay teacher |
| Tốc độ L40, 30 ảnh q3 | Teacher 17.2–17.5 s; student 11.3–11.4 s, chưa tính load | Cần số L4 và tổng thời gian bài nộp |
| Nạp mạng hiện tại | Khoảng 17–21 s | Đóng gói đúng trọng số inference có thể là cải tiến lớn |
| Tất định | CPU/GPU fp32 lệch; kiểm L40 cùng server khớp | Chưa chứng minh giữa máy L4 và môi trường chấm |
| Đóng gói | Chưa có decoder submission hoàn chỉnh | Làm ngay, không đợi hết nghiên cứu |
| Lưu trữ | Student q2/q3/q4 tồn tại trên server, khoảng 24 MB/file | Sao lưu cả base tương ứng và manifest; 24 MB không phải toàn codec |
| Compute | Lần kiểm tra không thấy job train của guess03; dịch vụ khác giữ ~1.5 GB mỗi GPU | Kiểm tài nguyên mỗi lần chạy; không dừng dịch vụ của người khác |

Ước lượng thiếu 30–55 / 95 / 40 Elo so với đội nhất 2025 chỉ là tham chiếu thô. Không dùng mốc 2025 để tuyên bố đủ thắng 2027. Không so PSNR của các cấu hình khác cách padding hoặc bộ ảnh như cùng một thí nghiệm.

## 3. Luật và lịch cần dùng

Trang [tasks chính thức](https://clic2027.compression.cc/tasks/) hiện quy định: ba rate, đánh giá bởi người; decoder tối đa 4 GB; 2 CPU, RAM 12 GB, GPU L4 24 GB; nhóm 25% chậm nhất không đủ điều kiện nhận giải. Có thể dùng decoder khác nhau cho từng rate, nhưng phải đăng ký chúng trong validation. Decoder test phải khớp hash đã nộp. Đây là giới hạn mỗi bài nộp, không được mặc định chia tất cả trọng số thành một gói duy nhất.

[Lịch chính thức](https://clic2027.compression.cc/schedule/2/): validation mở 01/09/2026, đóng 01/03/2027; test 02–09/03/2027; whitepaper 16/03/2027; hạn theo AoE. Lịch nội bộ bên dưới dùng ngày Asia/Bangkok và nộp sớm để có dự phòng. Trang submit hiện dẫn tới đăng nhập; chưa xác nhận một upload của tài khoản mình chạy thành công.

[Devkit chính thức](https://github.com/clic-challenge/devkit) mô tả `decode`, `bs.zip`, đầu ra `images/*.png`, chạy không mạng, và cho phép đề nghị Docker image riêng. Vì vậy thiếu diffusers/compressai trong image mặc định không đồng nghĩa bắt buộc port toàn bộ trước lần nộp đầu. Cần chuẩn bị phương án môi trường riêng và đánh giá khả năng được chấp nhận; phương án torch thuần vẫn có lợi cho dung lượng và khả năng kiểm soát.

Các điểm phải xác nhận trước quyết định cuối: byte được server tính thế nào đối với image ZIP/header; phạm vi timing gồm những bước nào; cách gộp thứ hạng ba rate; giới hạn gói được đo theo đơn vị nào; môi trường riêng và trọng số bên thứ ba được chấp nhận ra sao. Dùng devkit, kết quả nộp thử và phản hồi BTC làm căn cứ. Không coi giả định trong v2 là luật.

## 4. Tiêu chí nội bộ để ra quyết định

Các ngưỡng sau là mục tiêu quản trị thí nghiệm, không phải quy định hay dự báo thứ hạng.

| Mặt đánh giá | Mục tiêu / tiêu chí |
|---|---|
| Hợp lệ | Giải đủ ảnh, đúng tên/kích thước, không mạng, đúng ngân sách byte và hash |
| Dung lượng | Mỗi gói nhắm ≤3.6 GB theo cách đếm của server, chừa chỗ cho code và runtime |
| RAM / VRAM | Nhắm RAM ≤10 GB; server L40: tổng VRAM ≤30 GB/GPU theo cập nhật của người dùng. Bài nộp L4 24 GB vẫn nhắm ≤20 GB. Chừa headroom cho context/dịch vụ; giảm microbatch, dùng accumulation/checkpointing nếu cần |
| Tốc độ | Mục tiêu tham vọng ≤20 s/30 ảnh, báo riêng decode và tổng từ tiến trình mới; điều chỉnh bằng dữ liệu L4/server |
| Sàng lọc chất lượng | ΔQ̂ ≥+0.05 trên tập phát triển ở cùng ngân sách mới đáng đánh giá sâu; không tự động loại cải tiến nhỏ nhưng gần như miễn phí |
| Đưa vào bản chính | Có cải thiện trên ảnh độc lập, không lỗi chữ nghiêm trọng, vượt kiểm tra tốc độ/dung lượng/tất định |
| Mục tiêu chất lượng tháng 1 | So baseline hiện tại: +0.20 / +0.55 / +0.25 Q̂ ở ba rate; đây là stretch target, phải được người chấm xác nhận |
| Dừng nhánh nghiên cứu | Hết ngân sách pilot mà không có điểm tốt hơn đường chất lượng–byte–thời gian hiện có thì dừng hoặc đổi giả thuyết |

Không dùng ngưỡng 20 s làm bảo đảm thoát nhóm chậm nhất: ngưỡng loại thực phụ thuộc các bài dự thi khác. MS-ILLM cũng phải đo L4; không tự coi nó là phương án nhanh hơn DiT-IC.

## 5. Gói công việc P0: biến repo thành hệ thống có thể nộp (15–27/10)

### P0.1 — Khóa baseline và bảo toàn dữ liệu (15–17/10)

- Ghi Git SHA local/server, commit từng repo ngoài, SHA256 checkpoint, môi trường, seed, cấu hình padding và Q̂ hash.
- So sánh code đang chạy trên server với repo; đưa mọi thay đổi có ích về repo trước khi phát triển tiếp.
- Sao lưu `students`, các checkpoint base/LoRA cần tái dựng student, snapshot được chọn, stream kiểm tra, log và metrics ra nơi thứ hai. Kiểm hash và thử phục hồi một checkpoint.
- Tạo manifest kết quả ngày 13/10; giữ baseline bất biến. Q̂ bắt buộc được truyền rõ, thiếu Q̂ phải báo lỗi thay vì tự chuyển sang −LPIPS.
- Mỗi run mới có config, log, thời gian, checkpoint, CSV theo ảnh và summary; không ghi đè thư mục cũ.

**Xong khi:** từ manifest có thể tái dựng baseline, xác định đúng trọng số và phục hồi artifact. Không cần chạy lại toàn bộ bake-off cũ nếu checksum và một phép kiểm mẫu đã đủ xác nhận.

### P0.2 — Đo L4 và tất định (17–23/10)

- Đóng gói kit từ code hiện tại; kit đang có chỉ bao phủ q3, cần mở rộng q2/q4 và MS-ILLM trước quyết định cuối.
- L4 A: kiểm self-decode, L40→L4, thời gian load/unpack/entropy/DiT/DC-AE/ghi PNG, tổng wall time, RAM và VRAM.
- L4 B khác máy vật lý: giải stream từ A và tự tạo stream để A kiểm chiều ngược. Lưu GPU UUID, driver, torch/CUDA/cuDNN, CPU, thread count, hash container.
- Mỗi cấu hình đo ít nhất ba tiến trình mới, trong giới hạn 2 CPU và RAM tương ứng. Báo median, min–max, cold và warm riêng. Đo toàn bộ 30 ảnh và các kích thước lớn/lẻ trên tập độc lập.
- Cuối cùng đo hỗn hợp codec thực tế cho từng rate, gồm chi phí đổi/nạp model và VTM-SCC. Không suy ra tốc độ mix bằng trung bình số giây/ảnh của các run riêng.

**Xong khi:** có bảng L4 thật cho teacher/student/MS-ILLM/mix và kiểm index/symbol entropy bằng 0 lỗi. Khớp bitstream entropy và hash PNG là hai phép kiểm riêng; báo cả hai. Hai máy L4 cùng stack vượt kiểm tra vẫn cần xác nhận trên server CLIC.

**Nếu thất bại:** cô lập stage gây lệch và kiểm replay bằng stream cố định; cân nhắc fp64 chỉ cho phần gây lệch hoặc đường entropy số nguyên/QAT. Không chuyển toàn codec sang fp64 rồi bỏ qua chi phí. Hạ tầng int8 trong repo là điểm khởi đầu, chưa chứng minh thay thế được context model của DiT-IC/MS-ILLM.

### P0.3 — Có baseline submission sớm (17–27/10)

- Tạo `clic/submission/` với entrypoint `decode`, định dạng stream có version/mode/q/shape và metadata cần thiết; quy định padding, rounding, màu và tên file rõ ràng.
- Tạo `clic/tools/check_submission.py`: unpack thật, decode từ tiến trình sạch, kiểm tên/kích thước/số ảnh, checksum, byte, RAM, thời gian và không mạng.
- Dùng VTM-SCC/VVC sẵn có làm lần nộp kiểm tra đường đi; sau đó một checkpoint learned codec. Mục đích đầu tiên là kiểm định dạng và môi trường.
- So sánh hai phương án: đóng gói runtime đã có hoặc chuẩn bị image riêng; port torch thuần dựa trên `src/clic27/nn`. Chọn theo thời gian tới bản chạy được và kích thước thực, không dựa trên độ đẹp code.
- Nếu cần môi trường riêng, chuẩn bị Dockerfile, image digest và nội dung yêu cầu để chủ tài khoản gửi BTC. Trong phạm vi kế hoạch này chưa gửi email hay upload bài.
- Tạo 3 manifest gói theo bitrate. Đo tập trọng số tối thiểu: q3; q2+q3; q3+q4; q2+q3+q4. Loại checkpoint ít đóng góp nếu mất Q̂ nhỏ nhưng giảm dung lượng lớn.
- So sánh tensor giữa q: chỉ chia sẻ tensor thật sự giống nhau. Checkpoint đã merge LoRA không mặc nhiên dùng chung SANA. Delta dày vẫn có thể tốn gần bằng bản đầy đủ; kiểm kích thước file thật và parity sau tái dựng.

**Xong khi:** một submission validation được server chấp nhận, hoặc gói đã qua checker trong image đích và có lỗi truy cập server cụ thể cần giải quyết. Trạng thái thứ hai chưa được ghi là “đã nộp”.

## 6. Gói P1: sửa cách đánh giá trước khi train nhiều (16–31/10)

### Dữ liệu

- Giữ 30 ảnh CLIC làm benchmark tương thích lịch sử; không xem nó là tập kiểm định độc lập sau rất nhiều lần chọn model.
- Tạo bộ riêng 180 ảnh đề xuất: 100 ảnh phát triển + 80 ảnh giữ kín; bao phủ ảnh tự nhiên, người/khuôn mặt, chữ/UI, game, texture và cạnh sắc. Chia trước 80 ảnh giữ kín thành hai phần 40 ảnh cho mốc tháng 12 và tháng 1; phần đã mở không còn được gọi là kiểm định mới. Ghi rõ nguồn/quyền dùng và phân nhóm trước khi xem output.
- Loại trùng với train và 30 ảnh chính thức bằng hash và kiểm gần trùng; không chia crop của cùng ảnh vào hai tập.
- Train dùng DIV2K đang có, bổ sung CLIC train/LSDIR hoặc nguồn hợp lệ sau khi kiểm dữ liệu và dung lượng. Mở rộng có kiểm soát, không tải toàn bộ ngay.
- `train.py` chưng cất hiện đánh giá trên `clic_val`; bổ sung dev split độc lập để chọn checkpoint. Tập giữ kín chỉ mở ở mốc quyết định tháng, không dùng tìm seed hay checkpoint tốt nhất.

### Chấm mù

- Tuyển 3–5 người, dùng trang chấm hiện có; xoay trái/phải, có ảnh gốc, kết hợp toàn ảnh và crop cố định; thêm 10% câu lặp để kiểm độ nhất quán.
- Mỗi vòng lớn: 300 cặp nội dung, chia đều ba rate; mỗi cặp có ít nhất 3 phiếu, tương đương ≥900 quyết định. Chia thành phiên ngắn; có lựa chọn ngang nhau và ghi rõ cách tính.
- Bootstrap theo ảnh; xử lý việc cùng người chấm nhiều câu. Không coi crop và phiếu lặp từ cùng ảnh là các mẫu độc lập.
- Báo win rate, CI 95%, kết quả theo nhóm ảnh, và các lỗi nặng. Kỳ vọng finalist thắng baseline >55%; chỉ gọi bằng chứng rõ khi CI nằm trên 50%. Nếu CI còn rộng thì bổ sung mẫu hoặc giữ baseline.

### Q̂ và lỗi chữ

- Giữ v0 cố định để so lịch sử; kiểm bất đồng giữa Q̂ và người trên ảnh độc lập. Chỉ thay metric khi cải thiện ngoài tập đã dùng fit.
- Không dùng tỷ lệ người tự đồng ý 7/10 như “trần” để so trực tiếp với accuracy 81.2% khác protocol.
- Xây gate phía encoder cho nội dung chữ: dùng detection/OCR để đề xuất vùng, so tính dễ đọc và lỗi với bản gốc; không route toàn bộ ảnh screen sang SCC chỉ vì nhãn.
- Sinh ứng viên DiT/MS-ILLM/SCC rồi phạt lỗi chữ khi chọn mode. CER chỉ là tín hiệu phụ, phải kiểm crop bởi người do OCR cũng có lỗi.
- Báo cả ảnh tệ nhất và phân vị thấp; student q3 từng giảm tới 0.981 Q̂ ở một ảnh, nên trung bình không đủ.

## 7. Gói P2: nâng chất lượng trên 2 × L40 (cuối tháng 10–tháng 12)

### Nhánh R — mở rộng DiT-IC, ưu tiên 0.15

**Giả thuyết:** q4 bị giới hạn rate/codec; thêm bitrate có thể cải thiện chi tiết đủ để thắng mix hiện tại ở 0.15. Đây chưa phải kết luận.

1. Kiểm code train upstream, loss, tham số được cập nhật, checkpoint/LoRA chưa merge và license. `experiments/ditic_distill/train.py` hiện chỉ train đuôi decoder, không phải trainer rate–distortion.
2. Dựng trainer tối thiểu và tái hiện q4 trên tập nhỏ. Kiểm gradient, khả năng resume, entropy update và encode/decode thật trước run dài.
3. Đo trần biểu diễn của **đúng VAE/decoder đã fine-tune đang dùng**, ở cùng precision/padding. Nếu trần chi tiết ở bitrate cao thấp, tăng rate đơn thuần có thể không giải quyết được.
4. Đóng băng DiT/DC-AE, fine-tune phần codec với 3 hệ số rate-loss quanh cấu hình gốc. Chọn hướng hệ số sau khi đọc công thức: `R + λD` và `λR + D` cho chiều điều chỉnh ngược nhau.
5. Nhắm điểm thực quanh 0.12/0.15/0.20 bpp trên dev để allocator có lựa chọn; đo stream thật, không lấy likelihood làm bitrate cuối.
6. Chỉ khi có tín hiệu tốt mới mở thêm LoRA/adapter của generative decoder; giữ kiến trúc một bước và kiểm tăng dung lượng/tốc độ.
7. Sau khi cải thiện 0.15 được xác nhận, mở rộng 0.22/0.30/0.36. Không đầu tư ngang nhau cho cả ba rate ngay từ đầu.

**Dừng:** sau ba pilot rate và một pilot adapter, không có điểm Pareto tốt hơn MS-ILLM/mix trên ảnh độc lập thì giữ DiT cho rate thấp; chuyển compute sang nhánh M. Không cố cứu vì đã tốn thời gian.

### Nhánh M — cải thiện MS-ILLM ở 0.15 và 0.3

- Tái hiện fine-tune từ pretrained, kiểm metric/bitstream. Repo hiện có adapter bake-off, chưa có trainer MS-ILLM hoàn chỉnh.
- Giữ rate-loss và data cố định, sweep nhỏ trọng số LPIPS; sau đó sweep GAN từ cấu hình thắng. So cùng số bước/data, kiểm actual bpp.
- Bắt đầu từ nhóm trọng số có tác động hẹp; chỉ mở entropy/analysis transform khi cần và phải chạy lại kiểm tất định.
- Bổ sung dữ liệu cho nhóm thất bại như texture, tóc, mặt, chữ; mỗi thay đổi data là một thí nghiệm riêng.
- Q̂ làm tiêu chí chọn checkpoint; chưa tối ưu trực tiếp toàn Q̂ nếu chưa kiểm gradient, chi phí và nguy cơ khai thác sai metric.

**Dừng:** hai cấu hình perceptual không vượt baseline ngoài tập đã tune thì ưu tiên encoder search thay vì sweep rộng GAN.

### Nhánh D — tốc độ với mất chất lượng có kiểm soát

- Dùng student hiện tại làm baseline. So continued training với cùng lịch LR/steps để phân biệt lợi ích của loss mới với chỉ train thêm.
- Thử loss tới ảnh gốc thay vì chỉ tới teacher, với latent và ảnh ghép đúng; đây là thay đổi code/dữ liệu chưa có trong trainer hiện tại.
- Thử trọng số LPIPS nhỏ quanh cấu hình hiện tại. Không lặp lại GAN 0.3 đã cho kết quả kém nếu không có giả thuyết mới.
- Chỉ chạy nhánh D sâu nếu số L4 cho thấy cần. Ưu tiên giảm load, compilation hợp lệ và đóng gói trước khi hy sinh thêm chất lượng.
- AOTInductor/compile là thử nghiệm triển khai: đo build, cold-start, dynamic shapes, kích thước artifact, parity và tương thích đúng image. Không suy từ tốc độ warm trên L40.

## 8. Ma trận pilot và ngân sách

Mỗi run chỉ đổi một yếu tố so với parent. Các mức hệ số cuối cùng được điền sau bước reproduce; tránh đặt con số loss vô nghĩa khi chưa đọc trainer. Cùng seed 0, split/hash dữ liệu, số bước, batch hiệu dụng và lịch LR trong mỗi phép so. Ngân sách giờ là trần sơ bộ, không phải thời lượng đã đo.

| ID | Parent / đối chứng | Yếu tố thay đổi | Số run | Trần GPU-giờ/run | Kỳ vọng cần kiểm |
|---|---|---|---:|---:|---|
| R0 | q4 pretrained | Tái hiện train gốc | 1 | 8 | Không suy giảm ngoài sai số |
| R1 | R0 | Hệ số rate-loss: 3 mức theo chiều đã xác minh | 3 | 8 | Tạo điểm thực quanh 0.15 |
| R2 | R1 tốt nhất | Mở adapter generative decoder | 1 | 16 | Tăng chi tiết ở cùng byte |
| R3 | Cấu hình R thắng | Hệ số rate-loss cao-rate: 3 mức | 3 | 8 | Phủ vùng 0.3, chỉ chạy nếu R qua gate |
| M0 | MS-ILLM pretrained | Tái hiện fine-tune gốc | 1 | 8 | Đối chứng train hợp lệ |
| M1 | M0 | Trọng số LPIPS: 2 mức | 2 | 8 | Cải thiện ở 0.15 hoặc 0.3 |
| M2 | M1 thắng | Trọng số GAN: 2 mức | 2 | 8 | Texture tốt hơn, không phá mặt/chữ |
| D0 | Student q3 30k | Continued train cùng loss | 1 | 8 | Đối chứng cho train thêm |
| D1 | D0 cùng điểm khởi tạo | Đích loss: teacher → ảnh gốc | 1 | 8 | Giảm khoảng cách cảm nhận |
| D2 | D0 cùng điểm khởi tạo | Trọng số LPIPS: 2 mức | 2 | 8 | Cải thiện chất lượng không đổi tốc độ |

Tối đa **17 pilot, 144 GPU-giờ**. Đây là giới hạn danh mục, không chạy tất cả vô điều kiện. Thử smoke 200–500 bước để đo step time/VRAM; chốt số bước so sánh trong mỗi nhóm sao cho vừa trần giờ. Không so các run bị dừng ở số bước khác nhau như cùng ngân sách dữ liệu.

| Khoản | Ngân sách đề xuất |
|---|---:|
| 17 pilot tối đa | 144 GPU-giờ |
| Đánh giá, tạo stream, kiểm parity | 32–64 GPU-giờ |
| Đưa hai nhánh tốt nhất lên run dài | 2 × 48 = 96 GPU-giờ |
| Hai lần lặp khác seed cho finalist đã chọn, cùng số bước với run dài | 2 × 48 = 96 GPU-giờ |
| Tổng vòng nghiên cứu đầu | **368–400 GPU-giờ** |
| Dự phòng entropy QAT nếu cần | Tối đa 32 GPU-giờ pilot, ước lượng lại trước full train |

Với hai GPU dành trọn thời gian, trần lý thuyết là 336 GPU-giờ/tuần. Kế hoạch lấy **200 GPU-giờ/tuần** làm công suất làm việc ban đầu để chừa đánh giá, lỗi, tải máy và thời gian triển khai; vòng đầu dự trù 2–3 tuần lịch sau khi trainer sẵn. Kiểm lại throughput và quyền sử dụng server trước khi cam kết lịch.

GPU 0 ưu tiên R; GPU 1 ưu tiên M, eval và D. Không chạy benchmark tốc độ khi máy có tải gây nhiễu. Bắt đầu bằng hai job một GPU độc lập; chỉ dùng DDP khi đo chứng minh có lợi hoặc model không vừa một GPU.

L4: dự trù ban đầu 4–8 GPU-giờ qua ít nhất hai máy vật lý; tổng chiến dịch 12–24 giờ cho các vòng kiểm lại. Đây là hạn mức giờ đề xuất, chưa phải mua/thuê. Chi phí tiền = báo giá lúc thuê × số giờ thực, cộng lưu trữ/truyền dữ liệu nếu có. Không cần API trả phí cho kế hoạch này.

Lưu trữ dự trù 200–400 GB gồm data/cache/recon/weights; đo `du` trước khi tải thêm. Mỗi run giữ last, best và tối đa vài snapshot; checkpoint inference đầy đủ khác snapshot 24 MB. Giữ bản sao thứ hai của artifact không thể tái tạo rẻ.

## 9. Gói P3: lợi thế ở encoder và allocator (tháng 11–tháng 1)

- **Chia bit theo stream thật:** giữ DP hiện có, thêm header/mode/seed/shape và overhead ZIP theo cách server tính. Đo gói cuối; giữ biên 0.5–1% ban đầu, chỉ thu hẹp khi accounting đã khớp.
- **Thêm ràng buộc deploy:** allocator phải biết tập model đã đóng gói, chi phí load một lần, thời gian mỗi mode và memory. Chạy lại allocation sau khi bỏ q hoặc đổi teacher/student.
- **Không thêm codec chỉ vì một ảnh:** bỏ CoD-Lite hoặc một mức q, đo lại mất Q̂ so với chi phí dependency, tải model và kích thước. Chọn gói tối giản trên đường Pareto.
- **Seed search:** chỉ dùng nếu decoder cho phép tái tạo seed xác định; 1→4→8 seed là các ablation riêng. Seed phải được truyền và tính byte. Sàng lọc trên dev, không tune metric theo holdout.
- **Latent optimization:** tối ưu phía encoder với decoder cố định; đánh giá bpp thật sau quantization và entropy coding. Dừng nếu lợi ích chỉ có ở latent float hoặc mất khi mã hóa thật.
- **Chữ/khuôn mặt:** tối ưu chất lượng theo vùng để chọn candidate toàn ảnh trước. Chưa thêm codec theo patch/residual đa tầng trừ khi whole-image routing đã chứng minh không đủ.
- **Tốc độ encoder:** ghi thời gian mỗi ảnh/candidate; diễn tập toàn bộ test với số seed dự kiến. Chọn phương án hoàn thành ≤48 giờ trên compute thực để còn kiểm và nộp lại.

## 10. Lịch thực hiện và sản phẩm phải có

| Mốc nội bộ | Sản phẩm cần có | Quyết định |
|---|---|---|
| 15–17/10 | Manifest baseline, backup kiểm hash, danh sách code server lệch repo | Đủ khả năng phục hồi |
| 18–23/10 | Bảng L4 A/B, byte-accounting, skeleton submission, dev split | Chọn hướng tất định và môi trường |
| 24–27/10 | Baseline submission đầu, gói theo rate, shortlist teacher/student | Giữ tối đa 2 nhánh chất lượng + 1 nhánh tốc độ |
| 28/10–10/11 | Reproduce trainer R/M, pilot rate 0.15, chấm mù baseline mở rộng | R có tiềm năng thật hay chuyển ngân sách sang M |
| 11–30/11 | Run dài cho nhánh thắng, routing chữ, learned submission đủ ba rate nếu portal hoạt động | Có bản hợp lệ làm đường lui |
| 01–15/12 | Cải thiện 0.3, encoder search, kiểm trên holdout lần 1 | Loại nhánh không cải thiện ngoài dev |
| 16–31/12 | Tất định bản mới, gói ≤mục tiêu, timing mix trên L4 | Chọn tối đa 2 finalist |
| 01–15/01/2027 | Cả ba rate được server validation chấp nhận, holdout lần 2, chấm mù đủ người | Mốc bắt buộc nội bộ |
| 16–31/01 | Tối ưu tốc độ/load, sửa lỗi nhóm ảnh yếu, encoder rehearsal | Hạn cuối thêm kiến trúc lớn |
| 01–14/02 | Freeze kiến trúc; chọn checkpoint, seed policy, manifest; whitepaper nháp | Không mở nhánh nghiên cứu mới |
| 15–22/02 | Build sạch, nộp lại đủ rate, lưu hash và image digest được chấp nhận | **Khóa decoder nội bộ 22/02** |
| 23–28/02 | Kiểm tải xuống đúng artifact đã nộp, diễn tập phục hồi và encode test giả | Bản cuối sẵn trước hạn chính thức |
| 02–07/03 | Encode test, kiểm byte/output bằng decoder đã khóa, nộp sớm | Chỉ thay lựa chọn encoder trong khả năng decoder |
| 08–09/03 | Dự phòng lỗi upload/bitstream | Hoàn thành trước hạn test |
| 10–14/03 | Hoàn thiện whitepaper từ manifest, kết quả và code cuối | Nộp trước hạn 16/03 |

Whitepaper phục vụ điều kiện dự thi và tái tạo kết quả; không cần mở thêm hướng novelty. Nếu môi trường hoặc lịch chính thức đổi, cập nhật kế hoạch với ngày kiểm chứng mới.

## 11. Bản đồ thay đổi trong repo

Các đường dẫn ghi “mới” là hạng mục cần xây, chưa tồn tại hoặc chưa hoàn chỉnh tại ngày lập kế hoạch.

| Khu vực | Công việc |
|---|---|
| `experiments/l4_kit/` | Mở q2/q4/MS-ILLM, kiểm hai máy, đo tổng pipeline và giới hạn tài nguyên |
| `experiments/bakeoff/` | Byte accounting cuối, ràng buộc tập decoder/thời gian, metric thiếu phải fail rõ |
| `experiments/ditic_distill/` | Dev split riêng, paired original targets nếu thử D1, run metadata/resume |
| `experiments/ditic_finetune/` — mới | Trainer R tối thiểu sau khi đối chiếu upstream |
| `experiments/msillm_finetune/` — mới | Trainer M và export inference |
| `experiments/rating/` | Nhiều người, randomization, tie, kiểm lặp, tổng hợp theo ảnh |
| `src/clic27/nn/` | Parity DC-AE/SANA nếu chọn port; bổ sung codec/prompter/scheduler cần thiết |
| `src/clic27/entropy/` | Adapter tất định cho model được chọn; QAT chỉ khi cần |
| `clic/submission/` — mới | Decode, format version, loader, export, manifest và artifacts từng rate |
| `clic/encoder/` — mới | Sinh candidate, seed/latent search, routing, gói stream cuối |
| `clic/tools/` — mới | Checker, hash, budget, tài nguyên, diễn tập submission |
| `results/`, `reports/` | CSV/JSON/summary nhẹ; artifact lớn giữ ngoài Git và có backup |

Không làm dashboard/experiment framework mới nếu manifest + CSV + script hiện có đáp ứng được. README cần cập nhật server 2×L40 và liên kết tới kế hoạch này khi bắt đầu triển khai.

## 12. Quy trình chạy và phân tích

### Lệnh hiện có để bắt đầu kiểm tra, chưa khởi chạy bởi kế hoạch

Chạy từ root repo trên server, trong venv đã được kiểm tra phụ thuộc. Các lệnh dưới dùng CLI đã có; không giả lập lệnh trainer R/M chưa viết.

```bash
# Đọc tài nguyên và kiểm artifact trước mọi job.
nvidia-smi
ls -lh ~/clic/distill/students/{q2,q3,q4}.pt
sha256sum ~/clic/distill/students/{q2,q3,q4}.pt

# Đóng gói lại kit q3 bằng mã trong checkout hiện tại.
bash experiments/l4_kit/make_kit.sh \
  "$PWD/experiments" "$HOME/clic/distill/students" \
  "$HOME/clic/clic27_l4/out_server-01/streams_server-01.pt" \
  "$HOME/clic/clic27_l4_kit_20261015.tgz"

# Chấm lại student q3 vào thư mục mới; cần venv eval và các path này tồn tại.
CUDA_VISIBLE_DEVICES=0 python experiments/ditic_distill/eval.py \
  --q q3 --lat "$HOME/clic/distill/lat_q3/clic_val" \
  --ckpt "$HOME/clic/distill/students/q3.pt" \
  --valid "$HOME/clic/work_full_gpu/bakeoff/valid" \
  --qhat "$HOME/clic/inputs/qhat_v0/qhat/qhat_v0.json" \
  --out "$HOME/clic/distill/eval_q3_verify_20261015"
```

Trên L4, sau khi tải và giải nén kit, CLI hiện có là `bash clic27_l4_kit/l4_kit/run_all.sh`; máy thứ hai truyền thêm đường dẫn stream từ máy thứ nhất. Kit này cài môi trường có dependency phục vụ đo, chưa phải gói submission offline cuối cùng.

Mẫu run metadata cần ghi, không phải config mà trainer hiện tại tự đọc:

```yaml
run_id: R1_rate_a
parent_run: R0
factor: rate_loss_coefficient
value: TBD_after_loss_audit
seed: 0
data_manifest_sha256: TBD
initial_checkpoint_sha256: TBD
git_commit: TBD
effective_batch_size: TBD_after_smoke
steps: TBD_same_within_comparison
max_gpu_hours: 8
evaluation: actual_bitstream_qhat_v0_and_human_shortlist
```

**Phân tích mỗi run:** actual bytes, Q̂ v0, LPIPS/DISTS/PSNR, lỗi chữ/mặt, decode/load/total time, RAM/VRAM, gói decoder; plot chất lượng–bpp và chất lượng–thời gian; bảng theo ảnh và CI bootstrap theo ảnh. So cùng ngân sách, cùng protocol; điểm q cố định dùng để chẩn đoán, allocation cuối dùng để quyết định.

Run crash/OOM ghi riêng, không tính như điểm chất lượng thấp và không im lặng loại ảnh. Run hoàn thành thiếu ảnh không được xếp hạng. Finalist phải chạy seed xác nhận; không lấy best seed làm ước lượng trung bình. Mọi thay đổi checkpoint, precision, entropy hoặc image đích đều kích hoạt kiểm lại stream và submission.

## 13. Quy tắc cắt lỗ và phương án dự phòng

| Vấn đề | Hành động |
|---|---|
| L4 chậm | Profile tổng trước; tối ưu load/export/compile; student hoặc bỏ mode đắt chỉ sau đo mix |
| Float entropy vẫn lệch | Thu hẹp phần tính tất định, thử fp64 cục bộ; QAT integer có ngân sách riêng; bỏ model nếu chưa có đường decode tin cậy |
| Gói quá lớn | Gói theo rate, giảm q, chia tensor giống thật, adapter chưa merge nếu có; chỉ lượng tử trọng số sau parity/quality eval |
| DiT high-rate không vượt M | Dừng R ở rate cao; giữ low-rate, chuyển compute sang M/encoder |
| Q̂ tăng nhưng người chấm không thích | Không promote; ghi failure case, sửa cách đánh giá trên tập độc lập |
| Không có thêm người chấm | Giảm mức tự tin; ưu tiên cải thiện rõ, tăng ảnh giữ kín và nhiều phiên chấm của chủ dự án |
| Mất server | Phục hồi từ backup; tái tạo runner và dùng gói baseline đã chấp nhận |
| Portal lỗi hoặc BTC chưa duyệt image | Giữ log lỗi; tiếp tục checker và phương án image mặc định; chủ tài khoản xử lý liên hệ |
| Không rõ license | Kiểm code và weights riêng, xác nhận điều kiện dùng trước khi chọn finalist; chưa có kết luận pháp lý từ báo cáo cũ |

Không mở lại S1+residual, SD-Turbo refiner, CRDR hoặc Cool-chic chỉ để có thêm run. Cool-chic đang tạm dừng theo quyết định trước. Chỉ mở lại khi có thay đổi cụ thể đủ mạnh để bác lý do đã loại.

## 14. Năm việc cần làm ngay, theo thứ tự

1. Khóa manifest và sao lưu checkpoint/base/stream, xác nhận code server khớp repo.
2. Đưa baseline qua đường submission thật, đồng thời chuẩn bị L4 A/B và chọn môi trường triển khai.
3. Tạo dev/holdout độc lập và vòng chấm mù có thêm người.
4. Dựng reproduce trainer DiT-IC q4 và MS-ILLM; dùng hai L40 cho hai nhánh pilot có giới hạn.
5. Chốt 27/10 các rủi ro triển khai; tới 10/11 quyết định có mở rộng DiT-IC lên 0.15 hay dồn sang MS-ILLM. Duy trì bản hợp lệ ở cả ba rate trong khi cải tiến.

### Nguồn nội bộ

- `reports/CLIC2027_tinh_hinh_repo_2026-10-14.md`
- `results/2026-10-13_bakeoff_v0/`
- `results/2026-10-11_ditic_distill.md`
- `results/2026-10-12_ditic_determinism/`
- `results/2026-10-13_l4_kit_l40/`
- Server: `~/clic/work_merge_1013_v0/summary.md`, `~/clic/distill/eval_q*_gan01_snap_30000/summary.md`, `~/clic/l4kit_tarball.log`, `~/clic/clic27_l4/out_server-01/`.

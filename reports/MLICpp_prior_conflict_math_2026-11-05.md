# Cơ sở toán học cho mã hóa hiệu chỉnh prior

**Có thể chứng minh các mệnh đề nền và một bảo đảm lượng tử hóa dưới giả thiết cụ thể. Chưa có chứng minh phương pháp mới thắng codec khác hoặc có tính mới cho CVPR.** Tài liệu này phân biệt phân bố thật với prior học được, sai số của decoder với thông tin bổ sung, và MSE với human preference.

Các chứng minh được viết đầy đủ ở đây để kiểm tra lập luận của [đề xuất nghiên cứu](../plans/MLICpp_scientific_nucleus_2026-11-05.md). Các phép phân rã conditional expectation, Gaussian rate–distortion và chặn Lipschitz là công cụ chuẩn, không nhận là định lý mới của dự án. Phản ví dụ là minh họa tự dựng, chưa audit để nhận novelty.

## 1. Thông tin và sai số decoder là hai nguồn cải thiện khác nhau

Cho $X\in\mathbb R^d$ là nguồn với $\mathbb E\|X\|_2^2<\infty$, $B$ là base stream và $C$ là thông điệp bổ sung. Mọi kỳ vọng trong mục này lấy theo **phân bố thật $P$**. Giả sử decoder gốc $g(B)$ có moment bậc hai hữu hạn. Đặt

$$
m_B=\mathbb E[X\mid B],\qquad m_{BC}=\mathbb E[X\mid B,C].
$$

**Mệnh đề 1.** Với $D_0=\mathbb E\|X-g(B)\|_2^2$ và $D_* =\mathbb E\|X-m_{BC}\|_2^2$, ta có

$$
\boxed{D_0-D_*=
\mathbb E\|m_B-g(B)\|_2^2+
\mathbb E\|m_{BC}-m_B\|_2^2.}
$$

**Chứng minh.** Viết $X-g(B)=(X-m_B)+(m_B-g(B))$. Tích chéo có kỳ vọng bằng 0 vì $\mathbb E[X-m_B\mid B]=0$. Do đó

$$
D_0=\mathbb E\|X-m_B\|_2^2+\mathbb E\|m_B-g(B)\|_2^2.
$$

Tiếp tục viết $X-m_B=(X-m_{BC})+(m_{BC}-m_B)$. Thành phần thứ hai đo được theo $(B,C)$, còn $\mathbb E[X-m_{BC}\mid B,C]=0$, nên tích chéo lại bằng 0. Suy ra

$$
\mathbb E\|X-m_B\|_2^2=D_*+\mathbb E\|m_{BC}-m_B\|_2^2.
$$

Thế vào đẳng thức trước được điều phải chứng minh.

Với **decoder thực tế** $h(B,C)$ có moment bậc hai hữu hạn, đặt $D_h=\mathbb E\|X-h(B,C)\|_2^2$. Cùng phép trực giao cho

$$
\boxed{D_0-D_h=
\underbrace{\mathbb E\|m_B-g(B)\|_2^2}_{\text{sai lệch decoder gốc}}
+\underbrace{\mathbb E\|m_{BC}-m_B\|_2^2}_{\text{thông tin bổ sung}}
-\underbrace{\mathbb E\|h(B,C)-m_{BC}\|_2^2}_{\text{sai lệch decoder mới}}.}
$$

**Hệ quả nghiên cứu:** thành phần đầu có thể giảm bằng một decoder tốt hơn mà không cần $C$. Vì vậy, không được gán mọi gain của một kiến trúc mới cho side bits. Cần đối chứng decoder-only mạnh. Ngược lại, nếu $g=m_B$, cải thiện chỉ còn đến từ thông tin bổ sung, trừ sai lệch của decoder mới.

Đây không phải bảo đảm MSE giảm cho mọi model học được: số hạng cuối có thể lớn. Các conditional means là đại lượng lý tưởng chưa biết; không thể ước lượng chính xác chúng chỉ từ một ảnh cho mỗi stream. Đẳng thức không tự chuyển sang LPIPS/DISTS hoặc preference.

## 2. Phản ví dụ: prior variance có thể chọn sai nơi dành bit

Xét hai tọa độ độc lập trong $[0,1]$, base stream cố định, và $0<\eta<1/2$. Phân bố thật là

$$
P(X_1=0)=P(X_1=1)=\tfrac12,\qquad
P(X_2=\tfrac12-\eta)=P(X_2=\tfrac12+\eta)=\tfrac12.
$$

Decoder gốc xuất $(1/2,1/2)$. Đây **chính là conditional mean thật**, nên không có bias MSE của point prediction. Sai số tổng hai tọa độ là $D_0=1/4+\eta^2$.

Nhưng prior học được $Q$ có bất định sai: đặt $\delta=\eta^2$ và

$$
Q(X_1=0)=Q(X_1=1)=\delta/2,\qquad Q(X_1=1/2)=1-\delta,
$$

còn $Q$ mô hình hóa $X_2$ đúng. Cả hai phân bố vẫn có mean $1/2$, nhưng

$$
\operatorname{Var}_Q(X_1)=\delta/4=\eta^2/4
<\eta^2=\operatorname{Var}_Q(X_2).
$$

So hai codec, mỗi codec dùng **đúng một bit bổ sung**:

- Codec A luôn gửi giá trị nhị phân của $X_1$. Decoder dùng bit đó khôi phục $X_1$ chính xác, giữ $\hat X_2=1/2$. Khi đó $D_A=\eta^2$.
- Codec B luôn gửi dấu của $X_2-1/2$. Decoder khôi phục $X_2$ chính xác, giữ $\hat X_1=1/2$. Khi đó $D_B=1/4$.

Tọa độ được khóa cố định trong từng codec và hai bên cùng biết mapping, nên không giấu chi phí index. Đây là mapping tái tạo đã nêu, **không phải posterior mean cập nhật bằng $Q$**.

**Mệnh đề 2.** Chính sách chọn tọa độ có variance lớn nhất theo $Q$ chọn codec B. Tỷ số sai số so với codec A là

$$
\boxed{\frac{D_B}{D_A}=\frac{1}{4\eta^2},}
$$

không bị chặn bởi một hằng số khi $\eta\downarrow0$, dù mọi giá trị nguồn đều trong $[0,1]$ và cả hai codec cùng dùng một bit. Chứng minh là các tính toán trực tiếp ở trên.

Ví dụ $\eta=0,01$: $D_A=0,0001$, $D_B=0,25$, tỷ số 2.500. Đây là **tỷ số trong toy example**, hoàn toàn không phải gain của model hoặc challenge.

**Phản ví dụ chỉ chứng minh:** variance của một prior bị misspecification không đủ để bảo đảm bit allocation tốt. Nó thể hiện **underdispersion của phân bố dự đoán**, không chứng minh bias của mean, hallucination ngữ nghĩa hoặc một hiện tượng phổ biến trên MLIC++.

**Các giới hạn quyết định:** nếu dùng posterior thật $P$, variance ranking chọn $X_1$ đúng. Ordinary residual magnitude cũng chọn $X_1$ trong ví dụ này. Do đó, phản ví dụ chưa phân biệt phương pháp prior-conflict với residual coding, và bắt buộc phải có đối chứng uncertainty đã hiệu chỉnh. Hiện tượng sai của uncertainty không được nhận là phát hiện mới chỉ từ toy construction.

## 3. Giới hạn bit trong một mô hình Gaussian

Giả sử thêm rằng phân bố **thật** thỏa $X\mid B=b\sim\mathcal N(m(b),\sigma^2 I_d)$ gần như chắc chắn, với $d\ge1$ và $\sigma>0$ cố định. Với mỗi $b$, $C$ được truyền bằng một mã prefix-free điều kiện, có độ dài bit trung bình $R<\infty$. Vị trí, mode, scale và metadata nguồn-cụ-thể phải nằm trong thông điệp tính phí. Xét decoder xác định $h(B,C)$ với sai số Euclidean tổng $D_h$.

**Mệnh đề 3.** Khi đó

$$
\boxed{D_h\ge d\sigma^2\,2^{-2R/d}.}
$$

**Chứng minh.** Bất đẳng thức mã prefix cho $H(C\mid B)\le R$, nên

$$
I(X;C\mid B)\le H(C\mid B)\le R.
$$

Nếu $D_h\ge d\sigma^2$, kết luận hiển nhiên vì $R\ge0$. Nếu $0<D_h<d\sigma^2$, dùng $\mathsf h$ cho differential entropy với log cơ số 2. Đặt $E=X-h(B,C)$; phép tịnh tiến theo điều kiện không đổi entropy, và Gaussian cực đại hóa entropy với covariance cho trước. Vì vậy

$$
\begin{aligned}
\mathsf h(X\mid B,C)&=\mathsf h(E\mid B,C)\\
&\le\mathsf h(E)\\
&\le\frac d2\log_2(2\pi e D_h/d).
\end{aligned}
$$

Do $\mathsf h(X\mid B)=\tfrac d2\log_2(2\pi e\sigma^2)$,

$$
R\ge I(X;C\mid B)
\ge\frac d2\log_2\frac{d\sigma^2}{D_h}.
$$

Giải theo $D_h$ được kết luận. Trường hợp $D_h=0$ đòi hỏi rate vô hạn cho nguồn Gaussian không suy biến, nên không xảy ra với $R<\infty$.

Đây là một converse Gaussian cổ điển, không là achievability result cho packet hữu hạn. Nguồn nền đã kiểm: Wiegand và Schwarz, *Source Coding: Part I of Fundamentals of Source and Video Coding*, §4.4.1, [bản tác giả/course PDF](https://web.stanford.edu/class/ee398a/BookWiegandSchwarz.pdf). Cách viết ở đây dùng $B$ có sẵn tại cả hai phía và **tổng** squared error của vector; công thức per-coordinate chia $D_h$ cho $d$.

Không thay $\sigma$ thật bằng scale dự đoán từ entropy model MLIC++. Không áp chặn Euclidean này cho trọng số PSD tùy ý, LPIPS, DISTS hoặc human preference. Mô hình Gaussian là giả thiết minh họa; chưa được xác nhận trên dữ liệu local.

## 4. Bảo đảm cho packet hệ số lượng tử hóa

Một kết quả gần triển khai hơn là kiểm soát phần lợi ích bị mất vì lượng tử hóa. Cố định một nguồn $x$ và stream $b$. Cho decoder hiệu chỉnh $G_b(c)$, $c\in[-A,A]^k$, với $A>0$, $k\ge1$. Giả sử $G_b$ liên tục và có hằng số Lipschitz $L_b$ trên hộp này theo Euclidean norm:

$$
\|G_b(c)-G_b(c')\|_2\le L_b\|c-c'\|_2.
$$

Do hộp compact, tồn tại $c^*$ tối thiểu hóa $\|x-G_b(c)\|_2$. Đặt $e^*=\|x-G_b(c^*)\|_2$. Dùng lượng tử hóa đều midpoint $2^r$ mức mỗi tọa độ, $r\ge1$, và truyền toàn bộ $k$ chỉ số, cộng $s$ bit metadata. Giả sử $A,k,r$ và codebook đã được hai bên biết; tham số nào chưa biết phải nằm trong $s$. Mọi header, index/location và alignment/container overhead của packet cũng phải nằm trong $s$, nên tổng packet là $kr+s$ bit.

**Mệnh đề 4.** Với hệ số đã lượng tử hóa $\hat c$, ta có

$$
\boxed{\|x-G_b(\hat c)\|_2\le e^*+L_b A\sqrt{k}\,2^{-r}.}
$$

**Chứng minh.** Mỗi tọa độ của $c^*$ cách midpoint gần nhất không quá $A2^{-r}$, nên $\|\hat c-c^*\|_2\le A\sqrt{k}2^{-r}$. Áp bất đẳng thức tam giác và giả thiết Lipschitz cho

$$
\|x-G_b(\hat c)\|_2
\le\|x-G_b(c^*)\|_2+\|G_b(c^*)-G_b(\hat c)\|_2
\le e^*+L_b A\sqrt{k}2^{-r}.
$$

Nếu chỉ tìm được $\tilde c$ bằng optimizer, thay $e^*$ bởi sai số thực $\|x-G_b(\tilde c)\|_2$; không giả vờ đã đạt optimum. Bình phương vế phải cho chặn squared error.

Để bảo đảm thắng một reference ở **cùng tổng byte**, phải có thêm

$$
e^*+L_b A\sqrt{k}2^{-r}<e_{\mathrm{ref}},
$$

trong đó $e_{\mathrm{ref}}$ là sai số của reference đã được mã hóa ở ngân sách bao gồm packet. Điều kiện này chưa được xác minh trên model. Chặn Lipschitz có thể quá lỏng; local finite differences không chứng nhận một global upper bound. Rescale hệ số không tạo lợi ích giả: nếu thay $c$ bởi $tc$ với $t>0$, bán kính tăng $t$ và hằng số Lipschitz tương ứng giảm $t$, tích $L_bA$ giữ nguyên.

Chặn này là hệ quả chuẩn của lượng tử hóa đều và Lipschitz continuity, không nhận là contribution mới. Nó giải thích các yếu tố cần đo: chiều code, độ nhạy decoder, precision và achievable reconstruction error.

## 5. Giới hạn của phát biểu “tốt hơn residual coding”

**Mệnh đề 5.** Cố định base decoder $g$. Với mọi decoder xác định $H(b,c)$ trong một không gian vector, tồn tại residual decoder

$$
R(b,c)=H(b,c)-g(b)
$$

sao cho $g(b)+R(b,c)=H(b,c)$ với cùng thông điệp $(b,c)$. Ngược lại, mọi residual decoder $R$ tạo một $H=g+R$.

**Chứng minh.** Thay định nghĩa $R$ vào tổng và rút gọn. Hai cách viết giữ nguyên mọi reconstruction, nên giữ nguyên mọi distortion chỉ phụ thuộc nguồn/reconstruction và độ dài thông điệp.

Do đó, không thể chứng minh ưu thế biểu diễn trước **mọi residual decoder có điều kiện, không giới hạn capacity**. Đây là tương đương hàm, không bảo đảm hai cách hiện thực có cùng latency hoặc numerical behavior. Một phát biểu có thể kiểm chứng phải chỉ rõ family đối chứng, chẳng hạn fixed-basis rank-$k$, transform residual với syntax cố định, hoặc neural residual cùng ngân sách model/runtime. Không giấu một prior mạnh hơn trong phương pháp mới rồi gọi phần chênh lệch là ưu thế tuyệt đối của “prior conflict”.

## 6. Một ưu thế có thể chứng minh: basis có điều kiện so với basis cố định

Cố định $g(b)=(1/2,1/2)$, $0<a\le1/2$. Cho $B$ đều trên $\{0,1\}$ và $S$ đều trên $\{-1,1\}$, độc lập. Gọi $e_1,e_2$ là hai vector đơn vị của $\mathbb R^2$ và đặt

$$
X=g(B)+aS e_{B+1}.
$$

$B$ đã có trong base stream, nên cả encoder và decoder biết hướng nào có thể chứa chi tiết. So hai family có một hệ số và **một side bit**:

- Conditional basis: $H_{\rm cond}(B,S)=g(B)+aS e_{B+1}$.
- Fixed rank-one basis: $H_{\rm fixed}(B,C)=g(B)+\alpha(B,C)u$, trong đó $\|u\|_2=1$ và $u$ không phụ thuộc $B$. Để lấy lower bound mạnh hơn, cho phép encoder/decoder chọn bất kỳ hệ số thực nào, không chỉ một bit.

**Mệnh đề 6.** Conditional basis đạt sai số tổng bằng 0 với một side bit. Mọi fixed rank-one basis có sai số kỳ vọng ít nhất $a^2/2$. Chặn này đạt được bằng một side bit.

**Chứng minh.** Conditional decoder khôi phục chính xác theo định nghĩa. Với fixed $u$, khoảng cách bình phương từ residual $aS e_{B+1}$ đến đường thẳng $\operatorname{span}(u)$ là

$$
a^2\bigl[1-(u^\top e_{B+1})^2\bigr].
$$

Không một scalar coefficient nào tạo sai số thấp hơn khoảng cách chiếu trực giao này. Lấy trung bình trên $B$, dùng $u_1^2+u_2^2=1$, được

$$
D_{\rm fixed}\ge\frac{a^2}{2}\bigl[(1-u_1^2)+(1-u_2^2)\bigr]=\frac{a^2}{2}.
$$

Chọn $u=(1,1)/\sqrt2$ và truyền $S$ bằng một bit. Hệ số $\alpha=aS/\sqrt2$ cho sai số $a^2/2$ ở cả hai giá trị của $B$, nên đạt chặn.

Điều được chứng minh là **conditioning giúp một family hạng thấp xác định**, không phải mọi residual codec, mọi nguồn hay hai neural network cùng số tham số. Một residual decoder có điều kiện cũng thực hiện được $H_{\rm cond}$, đúng như Mệnh đề 5. Chứng minh không cần giả định prior sai, nên tự nó chưa xác nhận hạt nhân prior-conflict.

Với $a=0,4$, ví dụ đạt $D_{\rm cond}=0$ và $D_{\rm fixed}=0,08$. Đây là toy model. Không chuyển hai con số thành dự báo MLIC++ hoặc human preference.

## 7. Điều kiện thắng ở cùng tổng bit

Chọn một operating point reference thực sự mã hóa được với tổng budget $R$ bit. Để dành $s$ bit cho packet hiệu chỉnh, phương pháp dùng một base operating point thực sự mã hóa được với budget $R-s$. Mọi model/rate identifier, length field, index, scale và padding phải được tính. Không giả định quality có thể nội suy liên tục theo $R$.

Gọi distortion của reference là $D_{\rm ref}(R)$, của reduced-rate base là $D_{\rm low}(R-s)$, và của base cộng packet thực là $D_{\rm method}(R-s,s)$. Cùng distribution nguồn và cùng distortion $d$ phải được dùng cho cả ba.

**Mệnh đề 7.** Đặt

$$
T=D_{\rm low}(R-s)-D_{\rm ref}(R),\qquad
\Gamma=D_{\rm low}(R-s)-D_{\rm method}(R-s,s).
$$

Khi đó $D_{\rm method}-D_{\rm ref}=T-\Gamma$. Phương pháp thắng reference khi và chỉ khi **gain hiệu chỉnh thực $\Gamma$ lớn hơn phần chất lượng mất do giảm base rate $T$**.

**Chứng minh.** Cộng rồi trừ $D_{\rm low}(R-s)$ trong $D_{\rm method}-D_{\rm ref}$.

$T$ không bị giả định không âm: operating points của các codec thực có thể không đơn điệu với metric đang xét. $\Gamma$ ở đây đã gồm hiệu ứng optimizer, quantization và decoder thực; không trừ thêm các sai số đó lần thứ hai. Nếu dùng oracle chưa lượng tử hóa, phải định nghĩa relaxed feasible set và riêng phần approximation gap trước khi dùng một chặn khác.

Một phiên bản hữu ích cho finite codebook là: giả sử mọi candidate $c\in\mathcal C_b$ đã serialize, feasible trong budget và có reconstruction $H(b,c)$. Đặt

$$
D_{\rm oracle}=\mathbb E\min_{c\in\mathcal C_{B_{\rm low}}}d(X,H(B_{\rm low},c)),\qquad
\Gamma_*=D_{\rm low}-D_{\rm oracle}.
$$

Nếu selector có regret kỳ vọng $\varepsilon_{\rm sel}=D_{\rm method}-D_{\rm oracle}\ge0$, thì

$$
\boxed{D_{\rm method}-D_{\rm ref}=T-\Gamma_*+\varepsilon_{\rm sel}.}
$$

Đây là đẳng thức, không là tuyên bố oracle tốt hoặc selector có regret nhỏ. Với $\mathcal C_b$ hữu hạn, không rỗng, và exact encoder search theo chính $d$, regret bằng 0. Nếu dùng utility predictor sai lệch tối đa $\epsilon$ trên mọi candidate feasible của cùng $(x,b)$, chọn argmax predicted gain có regret không quá $2\epsilon$.

**Chứng minh chặn selector.** Với gain thật $u$, predicted gain $\hat u$, oracle $c^*$ và lựa chọn $\hat c$, ta có

$$
u(c^*)\le\hat u(c^*)+\epsilon
\le\hat u(\hat c)+\epsilon
\le u(\hat c)+2\epsilon.
$$

Uniform error là giả thiết mạnh chưa được chứng nhận cho predictor trên ảnh mới. Encoder có thể search trực tiếp với metric tính được, nhưng exact search theo LPIPS không là oracle theo preference người xem. Nếu codebook chỉ gồm các candidate đã lượng tử hóa, quantization đã nằm trong oracle; không cộng lại một quantization penalty.

## 8. Thuật toán cụ thể và kiểm chứng hữu hạn

Một phiên bản codec có thể định nghĩa hoàn chỉnh trước khi chọn neural parametrization:

1. Freeze base codec, các model assets, generator $H_\theta(b,c)$ và tập chỉ số hữu hạn $\mathcal C$ trước khi nén test. Trọng số không chứa dữ liệu ảnh test. Candidate $c=0$ có thể là identity correction, nhưng vẫn chịu phí signaling của format nếu được dùng.
2. Encoder tạo base stream theo một operating point đủ chừa chỗ cho packet. Decoder dựng mọi điều kiện chỉ từ base và model assets chung.
3. Encoder dựng/serialize candidates và loại những candidate vượt **actual total byte budget**. Giữ nguyên index của codebook decoder đã biết; không đánh số lại riêng tập feasible theo nguồn nếu chưa truyền mapping. Nếu không có candidate hợp lệ, báo không khả thi hoặc chuyển sang operating point base khác đã định trước; không bỏ ảnh.
4. Trong tập feasible, encoder chọn chỉ số tối thiểu hóa distortion đã khóa. Với $K$ chỉ số dùng fixed-width packing, payload index cần $\lceil\log_2 K\rceil$ bit, cộng tất cả metadata/padding. Điều kiện/basis nguồn-cụ-thể chưa suy ra được từ base phải được truyền và tính phí.
5. Decoder đọc base và index rồi tính $H_\theta(b,c)$ một lần; không có ảnh nguồn hoặc optimization ở decoder. Encoder có thể chạy search nhiều lần; chi phí đó vẫn phải báo riêng.

Một parametrization để kiểm ở pilot là $H_\theta(b,c)=T(f_b+U_\theta(f_b)a_c)$, với $f_b$ lấy từ entropy-decoded latent, $T$ là synthesis tail đã khóa và $a_c$ là code vector. Đây là **conditional codebook/residual family**, chưa là một cơ chế prior-conflict đã được xác nhận. Muốn giữ tên prior-conflict phải bổ sung measurement có năng lực và chứng minh giá trị vượt residual/calibrated-uncertainty controls. Không thể tự gắn nhãn “sửa prior” chỉ từ một codebook có điều kiện.

Đã hiện thực finite toy encoder/decoder tại [innovation_math_checks.py](../experiments/mlicpp/innovation_math_checks.py). Nó kiểm cả ví dụ ở mục 2 và mục 6, tính bằng phân số chính xác, đọc reconstruction từ packet bytes mà không nhìn nguồn. Format toy gồm 7 byte header và 1 byte payload, tổng **8 byte/packet ở cả hai đối chứng của mỗi family**; B/sign/index và padding đều được tính. Đây là format minh họa, không phải stream MLIC++ hay gói CLIC.

Đã chạy 16 packet qua file round-trip, xác nhận sai số chính xác, kiểm phép phân rã với decoder không tối ưu, kiểm cả budget-tax win/loss, và từ chối bốn loại packet lỗi. Kết quả tại [summary.json](../results/2026-11-05_mlicpp_innovation_math/summary.json), byte cụ thể tại [packets.json](../results/2026-11-05_mlicpp_innovation_math/packets.json).

Chạy lại bằng output mới:

```powershell
python experiments/mlicpp/innovation_math_checks.py --out results/innovation_math_new_run
```

Kiểm midpoint quantization cũng minh họa rằng chặn lỗi giảm theo precision **không bảo đảm lỗi thực giảm đơn điệu ở mọi điểm**, vì các tập midpoint không nhất thiết nested. Không thêm một kết luận đơn điệu không có trong Mệnh đề 4.

## 9. Kết luận cho hướng nghiên cứu

**Cập nhật thực nghiệm 05/11:** [pilot trên 32 nguồn mới](MLICpp_innovation_pilot_2026-11-05.md) đã train và giải mã thật; conditional basis không vượt fixed control về LPIPS/DISTS. Ví dụ tồn tại ở Mệnh đề 6 không bảo đảm neural family học hữu hạn sẽ đạt lợi thế đó. Dừng claim conditioning cho recipe đã thử; không dùng toy theorem để giải thích ngược kết quả không ủng hộ giả thuyết. Pilot chưa đo prior confidence nên chưa kiểm trực tiếp giả thuyết prior-conflict.

Các chứng minh xác lập ba nghĩa vụ thực nghiệm: tách decoder mismatch khỏi information gain; kiểm uncertainty bị sai thay vì mặc định nó đúng; tính precision và metadata khi báo gain của correction. **Chưa chứng minh prior-conflict coding tốt hơn residual coding**, chưa có theorem về preference hoặc thứ hạng CLIC, và chưa có định lý mới để nhận contribution CVPR.

Muốn có kết quả toán học mới cần cố định kiến trúc/prior update, một lớp nguồn rõ ràng và một comparator thực sự mạnh; sau đó chứng minh một chặn không chỉ lặp lại conditional expectation hoặc rate–distortion chuẩn. Nếu kiểm nghiệm không cho thấy gain ngoài residual và calibration controls, bỏ hạt nhân, không dùng các mệnh đề nền để giữ một model không hiệu quả.

Kiểm tra bổ sung: phản ví dụ đã được tính bằng phân số chính xác; một nhánh audit đã kiểm độc lập phép phân rã, các giả thiết Gaussian, residual equivalence, fixed-rank construction và budget tax. Audit chỉ rõ toy thể hiện underdispersion chứ không phải mean bias và tránh trừ quantization/optimization gap hai lần. Audit cuối đọc chứng minh mục 5–8 và code, không tìm thấy lỗi trọng yếu; nhánh audit không tự chạy script. Phiên chính đã chạy 16 packet; script không kiểm bằng máy tính các mệnh đề tổng quát, Gaussian converse hoặc selector bound — các kết quả đó dựa vào chứng minh đã viết. Các kiểm này không phải formal verification bằng Lean/Coq. Phần hình thức đã khép kín cho các family/giả thiết nêu trên; phần prior-conflict trên ảnh thật và tính mới khoa học vẫn chưa được xác nhận.

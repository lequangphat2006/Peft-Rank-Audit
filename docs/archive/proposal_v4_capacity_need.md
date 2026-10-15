> **LƯU TRỮ — không còn hiệu lực.** Bản chuyển từ file Word `đề xuất bản 4 (Capacity need có phải là thuộc tính ổn định của module?)` (tác giả Lê Quang Phát, 2026-10-03) sang markdown, giữ nguyên nội dung để không mất chi tiết của hướng ban đầu. Đề tài đã đổi hướng ngày 2026-10-15, xem `docs/pivot_v2.md`. Không sửa file này; nếu có công thức hoặc bảng bị lệch khi chuyển định dạng, đối chiếu với file Word gốc.

# Đề xuất đề tài (bản 4): Capacity need có phải là thuộc tính ổn định của module?

Oct 3, 2026 · @Le Quang Phat

## 1. Tóm tắt và thay đổi so với bản 3

Bản 4 đổi câu hỏi từ "uncertainty có phải proxy tốt không" sang "capacity need của một lớp có đo được, ổn định và chuyển giao được không", và đặt một pilot làm cổng quyết định trước khi chi khoảng 420 giờ GPU.

**Tên đề tài (tiếng Việt):** Capacity need có phải là thuộc tính ổn định của module? Kiểm toán đa tác vụ, đa ngôn ngữ, nhận biết mất cân bằng lớp về phân bổ rank trong tinh chỉnh hiệu quả tham số.

**Tên đề tài (tiếng Anh):** Is Capacity Need a Stable Module Property? A Multi-Task, Multi-Lingual, Imbalance-Aware Audit of Rank Allocation in Parameter-Efficient Fine-Tuning.

**Tác giả liên hệ:** TS. Đinh Thị Hồng Huyên. **Đồng tác giả:** Lê Quang Phát. **Đơn vị:** Khoa Công nghệ Thông tin, Trường Đại học Quy Nhơn.

Đề tài không đề xuất thuật toán mới. Giá trị nằm ở phép đo, kiểm tra nhiễu và kiểm toán các giả định mà mọi phương pháp phân bổ rank đang dựa vào.

| Nội dung          | Bản 3                                       | Bản 4                                                                                          |
|-------------------|---------------------------------------------|------------------------------------------------------------------------------------------------|
| Câu hỏi trung tâm | Uncertainty có là proxy của capacity need?  | Capacity need có đo được vượt nhiễu, ổn định, chuyển giao được không; proxy là câu hỏi thứ hai |
| Đơn vị đo         | Chưa chốt (12 lớp hay 48/72 module)         | Lớp (12), kèm sweep theo loại module                                                           |
| Định nghĩa r\*    | Ngưỡng "FFT − δ" (dễ bị censored)           | Tham chiếu là rank cao nhất của chính lớp đó, cộng thước đo L_m                                |
| Nhiễu             | Chưa đo                                     | Pilot, phân rã phương sai lớp × rank so với seed, seed ghép cặp                                |
| Ngôn ngữ          | PhoBERT so với RoBERTa (confound với model) | XLM-R trên cả VI và EN; model đơn ngữ chỉ để nhân bản kết quả                                  |
| Mất cân bằng lớp  | Không có                                    | Thao túng có kiểm soát, N cố định                                                              |
| Khớp ngân sách    | Theo tổng rank, alpha cố định               | Khớp tham số, rank_pattern kèm alpha_pattern (α = 2r)                                          |
| Phạm vi           | 5 RQ trong một bài                          | Paper A (RQ1–RQ3); Paper B (calibration, Pareto)                                               |
| Ngân sách         | ~130 giờ GPU                                | ~420 giờ lõi, có cổng dừng sau pilot                                                           |

## 2. Bối cảnh và vấn đề nghiên cứu

Chúng ta đang phân bổ rank LoRA theo "nhu cầu dung lượng" của từng thành phần, nhưng chưa ai chứng minh nhu cầu đó là một đại lượng đo được và ổn định.

Các mô hình tiền huấn luyện (PhoBERT \[1\], ViSoBERT \[2\], XLM-R \[3\]) đạt kết quả cao trên tiếng Việt, nhưng tinh chỉnh toàn bộ tham số đòi hỏi VRAM vượt khả năng nhiều phòng nghiên cứu đại học. LoRA \[4\] dùng rank đều (ví dụ r = 8) cho mọi module. Các phương pháp thích ứng như AdaLoRA \[9\], IGU-LoRA \[10\], BaRA \[11\] và gần đây StatLoRA \[21\] cho phép rank khác nhau giữa các thành phần, dựa trên điểm số quan trọng (gradient, độ bất định, thống kê kiểm định).

Các phương pháp này tính điểm riêng cho từng task, nên việc capacity need phụ thuộc task không mâu thuẫn với chúng. Đề tài kiểm tra ba giả định ở tầng sâu hơn:

- **A1 (đo được).** Capacity need của một lớp đo được vượt nhiễu seed, đủ để xếp hạng các lớp một cách đáng tin.

- **A2 (dự đoán được).** Trong cùng một task, điểm số proxy (Fisher, uncertainty, ...) dự đoán được thứ hạng đó.

- **A3 (chuyển giao được).** Profile capacity need giữ được khi đổi task, ngôn ngữ, lượng dữ liệu, mức mất cân bằng. A3 chỉ cần cho các "bảng khuyến nghị rank tĩnh", nhưng là giả định ngầm của bất kỳ bảng nào như vậy.

Nếu A1 sai, A2 và A3 không có đối tượng để kiểm tra. Vì vậy A1 được kiểm tra trước, bằng pilot (mục 7).

## 3. Khoảng trống và định vị

Đề tài lấp ba khoảng trống về phép đo, không phải về thuật toán.

1.  **Thiếu định nghĩa formal.** Các bài về rank allocation dùng "importance score" nhưng không định nghĩa hay đo trực tiếp "capacity need".

2.  **Thiếu phép kiểm tra nhiễu.** Chưa rõ profile capacity need có vượt được biến thiên seed hay không, nên mọi tương quan với proxy đều có trần là độ tin cậy của chính profile.

3.  **Thiếu kiểm chứng đa chiều.** Task, ngôn ngữ, lượng dữ liệu và mất cân bằng lớp chưa được tách riêng. Mất cân bằng lớp đặc biệt đáng ngờ vì nó chi phối gradient, và gradient là nền của nhiều proxy.

**Tuyên bố định vị (an toàn).** Trong phạm vi rà soát của chúng tôi, chúng tôi không tìm thấy nghiên cứu nào kiểm chứng đồng thời (i) độ ổn định của profile capacity need qua seed, (ii) khả năng chuyển giao qua task và ngôn ngữ, (iii) ảnh hưởng của mất cân bằng lớp lên proxy. Rà soát phải hoàn tất, có ngày, và được ghi vào phụ lục trước khi nộp.

**Các công trình gần nhất cần đọc kỹ trước khi nộp.** IGU-LoRA (ICLR 2026) và BaRA (arXiv 6/2026) đã được xác minh tồn tại. Hai bài mới hơn là StatLoRA (arXiv 2607.20205, 7/2026), đặt phân bổ rank thành bài toán kiểm định giả thuyết, và RSLoRA (arXiv 2607.09757). Mình mới đọc phần tóm tắt của hai bài này, chưa đọc toàn văn. Cả hai đều là thuật toán mới, không phải kiểm toán, nên bổ sung chứ không thay thế đề tài. StatLoRA còn là ứng viên tự nhiên để đưa vào phần so sánh proxy hoặc kiểm tra downstream.

## 4. Câu hỏi nghiên cứu và giả thuyết đăng ký trước

Bảy giả thuyết trả lời bảy câu hỏi, xếp theo thứ tự phụ thuộc: nếu H1 không đứng, các giả thuyết sau không có đối tượng để kiểm tra. Mọi ngưỡng định lượng dưới đây được khóa sau pilot và trước khi chạy lưới chính.

| Mã        | Câu hỏi                                                                                                       | Giả thuyết                                                                        | Kiểm định                                                                                               |
|-----------|---------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| RQ1a / H1 | Capacity need có đo được vượt nhiễu seed không?                                                               | Phương sai tương tác lớp × rank lớn hơn 0                                         | Phân rã phương sai (moments) và kiểm định hoán vị nhãn lớp trong từng seed; mô hình hỗn hợp để xác nhận |
| RQ1b / H2 | Profile có ổn định qua seed không?                                                                            | Độ tin cậy split-half ≥ ngưỡng kỳ vọng                                            | Spearman–Brown trên hai nửa seed tách rời, CI bootstrap                                                 |
| RQ1c / H3 | Tác động của các lớp có cộng tính không?                                                                      | Profile đo cô lập dự đoán được F1 của các mẫu rank ngẫu nhiên (R² ≥ ngưỡng)       | Hồi quy F1 thực theo F1 dự đoán cộng tính, CI bootstrap                                                 |
| RQ1d / H4 | Profile có chuyển giao qua task, ngôn ngữ, lượng dữ liệu không?                                               | τ giữa hai cell ≥ 50% trần độ tin cậy của chính các cell đó                       | Kendall τb hiệu chỉnh suy giảm, bootstrap hai tầng                                                      |
| RQ2a / H5 | Proxy nào dự đoán được profile trong cùng task?                                                               | τb giữa proxy và profile \> 0,3 cho ít nhất một proxy                             | τb, top-k overlap; gộp các cell bằng meta-analysis hiệu ứng ngẫu nhiên                                  |
| RQ2b / H6 | Mất cân bằng lớp có làm giảm giá trị dự đoán của proxy không?                                                 | τb giảm khi ρ (tỷ lệ mất cân bằng) tăng, và Fisher cân bằng lớp phục hồi một phần | Hệ số dốc của τb theo log ρ, so sánh Fisher thường và Fisher cân bằng                                   |
| RQ3 / H7  | Phân bổ theo profile (oracle) hoặc proxy tốt nhất có vượt phân bổ đều và ngẫu nhiên, khi khớp tham số, không? | Có vượt; nếu không, "ngân sách quan trọng hơn cách phân bổ" là kết quả hợp lệ     | Wilcoxon ghép cặp theo seed, TOST cho tương đương                                                       |

Chú thích: ngưỡng 0,3 của H5 kế thừa bản 3; các ngưỡng của H2, H3, H4 được đặt sau pilot bằng mô phỏng power (mục 7). Một giả thuyết mà mô phỏng cho thấy không đủ power sẽ bị hạ xuống thăm dò trước khi chạy lưới chính, và việc hạ này được ghi vào kế hoạch thống kê.

## 5. Định nghĩa formal capacity need và các estimator

Capacity need của lớp m là rank nhỏ nhất mà tại đó hiệu năng của chính lớp đó gần như bão hòa, khi các lớp còn lại giữ rank cơ sở r_b = 8. Lấy đường cong của chính lớp làm tham chiếu (thay vì FFT − δ của bản 3) để r\* luôn tồn tại và không bị censored.

```latex
g_m(r) = \mathbb{E}_s\left[\mathrm{MacroF1}(\text{lớp } m \text{ có rank } r;\ \text{các lớp khác } r_b = 8)\right]
r^*_m(\delta) = \min\{\, r \in R : g_m(r) \ge g_m(r_{\max}) - \delta \,\}, \qquad R = \{1, 4, 16, 32\}
L_m = g_m(8) - g_m(1) \quad (\text{mức phạt khi bị "bóp" rank})
```

Ba đại lượng được báo cáo, theo thứ tự ưu tiên:

1.  **L_m (chính).** Hiệu năng mất đi khi bóp rank của lớp m từ 8 xuống 1. Đây là đại lượng có tín hiệu lớn nhất nên là profile chính dùng cho mọi kiểm định.

2.  **r\*\_m(δ) (thứ cấp).** Dùng đường cong được làm mịn bằng isotonic regression theo log2 r. δ được đặt sau pilot, bằng một bội số của sai số chuẩn ghép cặp, không phải một hằng số 0,01. Một lớp "phẳng" (mọi rank trong phạm vi nhiễu) được gắn nhãn riêng, và tỷ lệ lớp phẳng là một kết quả báo cáo.

3.  **AUC theo log2 r (thăm dò).** Diện tích dưới đường cong, không phụ thuộc ngưỡng.

**Ghép cặp.** Mỗi seed xác định cả split, khởi tạo và thứ tự dữ liệu. Với mỗi seed, cấu hình đều r = 8 được chạy một lần làm baseline chung, và mọi đại lượng được tính trên Δ = F1(cấu hình) − F1(baseline) cùng seed. Cách này triệt tiêu phần lớn nhiễu seed và là lý do lưới không chứa rank 8.

**Mô hình nhiễu.** Với Δ theo (lớp m, rank r, seed s), mô hình hỗn hợp và độ tin cậy được định nghĩa như sau.

```latex
\Delta_{m,r,s} = \alpha_r + u_{m,r} + \varepsilon_{m,r,s}, \quad u_{m,r} \sim \mathcal{N}(0, \sigma^2_{LR}), \quad \varepsilon \sim \mathcal{N}(0, \sigma^2_{\varepsilon})
\rho = \frac{\sigma^2_{LR}}{\sigma^2_{LR} + \sigma^2_{\varepsilon} / n_s}
```

H1 kiểm H0: σ²_LR = 0, bằng hoán vị nhãn lớp trong từng seed (các lớp trao đổi được dưới H0). H2 lấy ρ làm độ tin cậy của profile với n_s seed. ρ cũng là trần cho mọi tương quan với proxy, vì một proxy không thể dự đoán một profile tốt hơn chính độ tin cậy của nó.

**Giới hạn đã nêu trong hạn chế.** Đại lượng này có điều kiện theo ngữ cảnh (các lớp khác ở rank 8). Đây chính là điều H3 kiểm tra.

## 6. Thiết kế thực nghiệm

Lưới chính gồm bốn cell task × ngôn ngữ chạy trên cùng một mô hình (XLM-R-base), cộng ba cell phụ cho lượng dữ liệu, mất cân bằng lớp và loại module. Dùng XLM-R trên cả hai ngôn ngữ để hiệu ứng ngôn ngữ không bị lẫn với hiệu ứng mô hình.

| Cell | Task                                 | Ngôn ngữ | Dữ liệu    | Seed      | Số lượt |
|------|--------------------------------------|----------|------------|-----------|---------|
| C1   | Sentiment (UIT-VSFC)                 | VI       | 100%       | 5         | 240     |
| C2   | Topic (UIT-VSFC)                     | VI       | 100%       | 5         | 240     |
| C3   | Sentiment (SST-2, subsample)         | EN       | 100%       | 5         | 240     |
| C4   | Topic (AG News, subsample)           | EN       | 100%       | 5         | 240     |
| C5   | Sentiment (UIT-VSFC)                 | VI       | 50% và 20% | 3 mỗi mức | 288     |
| C6   | Mất cân bằng lớp (trên cell khả thi) | VI và EN | N cố định  | 3         | 432     |
| C7   | Sweep theo loại module (trên C1)     | VI       | 100%       | 3         | 72      |

Mỗi cell chạy 48 lượt mỗi seed (12 lớp × 4 rank {1, 4, 16, 32}); baseline r = 8 đều chạy thêm một lần mỗi seed. UIT-VSMEC là cell tùy chọn cho phân tích chuyển giao (3 seed). PhoBERT-base (VI) và RoBERTa-base (EN) chỉ dùng để nhân bản trên C1 và C3 khi pilot và lưới chính đã ổn.

**Dữ liệu và phân vùng.**

- SST-2 và AG News được subsample phân tầng về cùng quy mô với UIT-VSFC (~16 nghìn mẫu). Tập con cố định cho mọi seed.

- UIT-VSFC sentiment dùng split chia sẵn theo group_id: train 12.780, hiệu chuẩn (val) 1.598, test 1.599; lớp \[âm, trung tính, dương\] ở train là 5.986 / 522 / 6.272. Không có group_id hay văn bản trùng giữa các split; 3,5% mẫu val và 2,6% mẫu test có bản gần trùng (cosine ≥ 0,95) trong train và được nêu trong hạn chế. Split cố định cho mọi seed; seed chỉ đổi khởi tạo head, dropout và thứ tự dữ liệu, nên nhiễu seed không gồm biến thiên do chọn tập huấn luyện. SST-2 và AG News chia 80/10/10 trên tập con cùng quy mô.

- Max length 128, chi tiết như bản đặc tả triển khai.

**Cấu hình LoRA.** Mỗi lớp gắn cùng một rank cho 6 module: query, key, value, attention.output.dense, intermediate.dense, output.dense, được liệt kê đầy đủ theo tên (không dùng hậu tố "dense" để tránh trúng head phân loại). Rank theo lớp dùng rank_pattern, kèm alpha_pattern với α = 2r để hệ số co giãn α/r không đổi. Mức ngân sách được khớp theo tổng rank: ở mức lớp, mỗi đơn vị rank tốn cùng số tham số ở mọi lớp, nên khớp tổng rank cũng là khớp tham số (có kiểm tra bằng test đơn vị). Số epoch cố định, chốt ở pilot, không dùng early stopping trong sweep để rank không bị lẫn với điểm dừng.

**Mất cân bằng lớp có kiểm soát (C6).** Quy mô N huấn luyện được giữ bằng nhau giữa mọi mức. Tỷ lệ ρ = n_max / n_min ∈ {1, 5, 20}, số mẫu mỗi lớp theo cấp số nhân: n_k tỷ lệ với ρ^(−k/(K−1)), k = 0..K−1. Test set giữ nguyên. Trước khi chạy phải đếm số mẫu từng lớp trên split thật: N tối đa bằng K nhân số mẫu của lớp nhỏ nhất trong train. Nếu lớp neutral của UIT-VSFC sentiment quá nhỏ để hỗ trợ ρ = 1 ở N đủ lớn, chọn cell khác (topic, hoặc AG News) làm cell chính của C6. Quyết định này thuộc cổng G0 (mục 7). Calibration split theo prior huấn luyện (cần cho Paper B).

**Kiểm tra tính cộng (H3).** Sinh 40 mẫu rank ngẫu nhiên khớp ngân sách (tổng rank 96), chạy 3 seed trên C1, rồi so F1 thực với F1 dự đoán bằng cộng các Δ đo lớp đơn lẻ. Tổng 120 lượt.

## 7. Pilot và tiêu chí quyết định

Pilot (~125 lượt, ~31 giờ GPU) quyết định có nên chạy lưới chính hay không, vì nó trả lời H1 trên một cell trước khi chi phần lớn ngân sách.

**Thiết kế pilot.** Cell C1 với XLM-R-base, 5 seed. Mỗi seed chạy baseline r = 8 đều, cộng 12 lớp × rank {1, 32} (24 lượt), tức 25 lượt mỗi seed. Thêm grid LR ∈ {1e-4, 2e-4, 3e-4} trên baseline và chốt số epoch cố định.

**Đo gì.** σ²_ε (nhiễu seed sau ghép cặp), σ²_LR (tương tác lớp × rank), độ tin cậy ρ với n_s = 5, tỷ lệ lớp có \|Δ\| vượt 2 sai số chuẩn, và mô phỏng power cho H2 đến H5 từ các phương sai đo được. Chỉ sau mô phỏng mới khóa các ngưỡng định lượng của mục 4.

| Cổng         | Câu hỏi                                                     | Quy tắc quyết định                                                                                                                                                                                                                                                     |
|--------------|-------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| G0 (dữ liệu) | Cell nào khả thi cho mất cân bằng ρ = 1, 5, 20 ở N cố định? | Đếm lớp trên split thật; chọn cell có N tối đa ≥ 3.000; nếu không cell nào đủ, giảm N và ghi vào hạn chế                                                                                                                                                               |
| G1 (đo được) | Profile lớp có vượt nhiễu không?                            | ρ ≥ 0,5 và LRT có ý nghĩa: chạy lưới chính. 0,2 ≤ ρ \< 0,5: chạy lưới chỉ với C1 và C3, tăng seed, chuyển sang AUC và đơn vị thô. ρ \< 0,2: không chạy lưới đầy đủ; chuyển sang đơn vị thô (nhóm lớp dưới/giữa/trên và loại module) với kiểm định tương đương và power |
| G2 (power)   | Mỗi giả thuyết có đủ power?                                 | Power mô phỏng \< 0,8: hạ xuống thăm dò, ghi trước khi chạy                                                                                                                                                                                                            |

Các mốc 0,5 và 0,2 của G1 là tạm thời. Chúng được xác nhận hoặc đổi sau mô phỏng power, trước khi chạy lưới chính, và việc đổi được ghi vào kế hoạch thống kê.

Không dùng "shuffle nhãn rồi sweep" làm null: khi nhãn bị xáo trộn, F1 về mức ngẫu nhiên và r\* không còn ý nghĩa. Null dùng là hoán vị nhãn lớp trong từng seed, chỉ làm kiểm tra phụ cho phân rã phương sai.

## 8. Proxy và kiểm toán rank

Mỗi proxy được tính ở mức lớp và so với profile L_m, với độ tin cậy ρ của chính profile làm trần. Một proxy đạt τb gần trần đó là tốt hết mức có thể, dù τb tuyệt đối còn thấp.

| Proxy                       | Định nghĩa ở mức lớp                                     | Tính khi nào                       |
|-----------------------------|----------------------------------------------------------|------------------------------------|
| Fisher                      | Tổng bình phương gradient của các ma trận đích trong lớp | Sau warm-up, trên split huấn luyện |
| Fisher cân bằng lớp         | Như Fisher, nhưng mỗi lớp nhãn đóng góp bằng nhau        | Sau warm-up, kiểm soát cho H6      |
| Phương sai MC-dropout       | Phương sai logit khi chỉ bật dropout ở lớp đó            | Sau baseline r = 8                 |
| Entropy attention           | Entropy trung bình của phân phối attention của lớp       | Sau baseline r = 8                 |
| Effective rank ΔW           | exp(−Σ p_i log p_i) trên giá trị kì dị                   | Sau baseline r = 8                 |
| Spectral norm ΔW            | ‖ΔW‖₂ trung bình các ma trận trong lớp                   | Sau baseline r = 8                 |
| Intrinsic dimension cục bộ  | Số chiều của không gian gradient của lớp                 | Sau warm-up                        |
| StatLoRA p-value (tùy chọn) | Thống kê kiểm định theo \[21\]                           | Nếu có mã nguồn đầy đủ             |

**Tránh nhiễu dùng chung.** Proxy được tính từ các lần chạy có seed tách rời với các seed tạo profile, để nhiễu của hai bên không tương quan giả tạo.

**Các phép so sánh.** Kendall τb, Spearman, top-k overlap (k = 3, 5), và τb hiệu chỉnh suy giảm bằng căn bậc hai của độ tin cậy. Với chỉ 12 lớp, kiểm định hoán vị từng cell có power thấp, nên kết quả được gộp qua các cell bằng meta-analysis hiệu ứng ngẫu nhiên và bootstrap hai tầng (seed rồi mẫu kiểm tra).

**Mất cân bằng lớp (H6).** τb của mỗi proxy được tính riêng ở từng mức ρ. Báo cáo hệ số dốc theo log ρ và hiệu ứng của việc cân bằng lớp khi tính Fisher. Lưu ý: profile L_m cũng được đo lại ở từng mức ρ, vì đích đo có thể đổi theo prior.

## 9. Kiểm tra downstream: Dynamic LoRA và đối chứng

Kiểm tra downstream trả lời câu hỏi "biết capacity need có giúp phân bổ tốt hơn không", và được thêm đối chứng oracle để tách câu hỏi này khỏi chất lượng của proxy.

**Phân bổ.** σ_m ∈ \[0, 1\] là điểm đã chuẩn hóa min–max qua 12 lớp. Rank theo lớp r_m = round(r_min + (r_max − r_min) · σ_m) với r_min = 1, r_max = 32, sau đó điều chỉnh bằng phương pháp phần dư lớn nhất cho đến khi tổng rank bằng 96 (= 12 × 8). Ở mức lớp, tổng rank bằng nhau cũng là số tham số bằng nhau.

| Đối chứng       | Cách sinh σ_m                                                  | Mục đích                                   |
|-----------------|----------------------------------------------------------------|--------------------------------------------|
| Đều             | σ_m hằng, r = 8                                                | Baseline                                   |
| Ngẫu nhiên      | σ_m ~ U(0, 1), mỗi seed một mẫu                                | Hiệu ứng chỉ do ngân sách                  |
| Đảo ngược       | 1 − σ_m                                                        | Kiểm tra chiều của proxy                   |
| Hoán vị         | Trộn σ_m của Dynamic                                           | Kiểm tra thứ hạng có ý nghĩa               |
| Oracle          | σ_m từ profile L_m, bỏ seed đang đánh giá (leave-one-seed-out) | Trần lý thuyết của việc biết capacity need |
| Dynamic (proxy) | σ_m từ proxy tốt nhất và từ Fisher                             | Giá trị thực dụng của proxy                |

Oracle quan trọng vì nó phân biệt hai trường hợp. Nếu oracle không vượt phân bổ đều, việc biết capacity need chính xác cũng không đủ để cải thiện hiệu năng. Nếu oracle vượt nhưng proxy không, vấn đề nằm ở proxy.

**Seed và đánh giá.** 10 seed trên C1 và 5 seed trên C2 đến C4, thêm FFT và LoRA r = 8 đều làm mốc. Macro-F1 là chỉ số chính, so sánh ghép cặp theo seed bằng Wilcoxon, và TOST cho kết luận tương đương với biên được khóa sau pilot.

## 10. Kế hoạch thống kê

Bảy giả thuyết được đặt vào một họ xác nhận duy nhất với hiệu chỉnh Holm (m = 7), và mọi giả thuyết thiếu power theo pilot bị hạ xuống thăm dò trước khi chạy lưới chính.

| Họ               | Nội dung                                   | Phương pháp                                                                 |
|------------------|--------------------------------------------|-----------------------------------------------------------------------------|
| Xác nhận (m = 7) | H1 đến H7                                  | Mô hình hỗn hợp, bootstrap hai tầng, Holm; Wilcoxon ghép cặp và TOST cho H7 |
| Thăm dò 1        | Ràng buộc r\*(δ), AUC, các proxy phụ       | Effect size và khoảng tin cậy, không hiệu chỉnh                             |
| Thăm dò 2        | Nhân bản trên PhoBERT, RoBERTa; cell VSMEC | Effect size và khoảng tin cậy                                               |

**Nguồn phương sai.** Bootstrap hai tầng: lấy lại mẫu seed, rồi lấy lại mẫu mẫu kiểm tra, dùng cùng chỉ số mẫu cho mọi cấu hình để giữ ghép cặp.

**Power.** Mô phỏng từ σ²_ε và σ²_LR đo ở pilot: sinh hàng ngàn profile 12 lớp ở các mức độ tin cậy khác nhau, ước lượng power cho τb giữa hai profile và giữa proxy với profile. Nếu power \< 0,8, báo cáo "chưa đủ bằng chứng để kết luận", không diễn giải thành "không có tín hiệu".

**Tương đương.** Biên TOST Δ_eq cho Macro-F1 được chốt một giá trị duy nhất sau pilot, và chỉ dùng cho H7. Kết luận "không có hiệu ứng" ở các giả thuyết khác phải dựa trên khoảng tin cậy và power, không dựa trên p \> 0,05.

**Seed.** Pilot dùng seed 100–104, không dùng lại trong lưới chính. Lưới chính dùng seed 42–51 cho C1 và 42–46 cho C2–C4; proxy dùng seed riêng (200–204) để tách rời với seed tạo profile.

**Cảnh báo.** Không so ECE giữa các tập khác prior. Phân tích hiệu chuẩn thuộc Paper B.

## 11. Ngân sách GPU và tiến độ

Phần lõi cần khoảng 2.200 lượt, ~420 giờ GPU, và phần lớn khoản này chỉ chạy sau khi pilot qua cổng G1. Con số dùng giả định 15 phút mỗi lượt với tập huấn luyện 12.780 mẫu (UIT-VSFC) và phải được đo lại ở pilot; lượt ở dữ liệu nhỏ hơn được nhân theo tỷ lệ quy mô.

| Hạng mục                                                     | Số lượt   | Giờ GPU  |
|--------------------------------------------------------------|-----------|----------|
| Pilot (C1, 5 seed, rank {1, 32})                             | 125       | 31       |
| Lưới chính C1–C4 (4 cell × 5 seed × 48)                      | 960       | 240      |
| C5: 50% và 20% dữ liệu                                       | 288       | 25       |
| C6: mất cân bằng lớp, N cố định                              | 432       | 22       |
| C7: sweep theo loại module                                   | 72        | 18       |
| Kiểm tra tính cộng (40 mẫu × 3 seed)                         | 120       | 30       |
| Downstream: 8 cấu hình × 25 seed (10 + 3 × 5)                | 200       | 50       |
| Seed riêng cho proxy (4 cell × 5 seed) và tính proxy         | 20        | 5        |
| **Tổng lõi**                                                 | **2.217** | **~421** |
| Nhân bản PhoBERT (C1) và RoBERTa (C3), 3 seed (có điều kiện) | 288       | 72       |
| Cell UIT-VSMEC (tùy chọn, 3 seed)                            | 144       | ~11      |

Với mức giá thuê T4 mà bản đặc tả triển khai ước tính (50–80 USD cho 130 giờ), phần lõi được khoảng 160–260 USD. Giá thực cần kiểm tra lại trước khi thuê. Trên một GPU chạy liên tục, phần lõi mất khoảng 18 ngày.

| Giai đoạn | Thời gian  | Nội dung                                                             | Cổng                          |
|-----------|------------|----------------------------------------------------------------------|-------------------------------|
| P0        | Tuần 1–2   | Pipeline, split group-aware, đếm lớp, test đơn vị cho ngân sách rank | G0                            |
| P1        | Tuần 3–4   | Pilot, mô phỏng power, khóa ngưỡng                                   | G1, G2                        |
| P2        | Tuần 5–10  | Lưới chính C1–C4                                                     | Kiểm tra giữa kỳ sau C1 và C3 |
| P3        | Tuần 9–13  | C5, C6, C7, kiểm tra tính cộng                                       |                               |
| P4        | Tuần 12–15 | Proxy, downstream                                                    |                               |
| P5        | Tuần 14–18 | Phân tích, nhân bản có điều kiện, viết bản thảo                      |                               |

Tổng khoảng 18 tuần cho đến bản thảo, cộng 6 đến 8 tuần chỉnh sửa và nộp, tức khoảng 6 đến 7 tháng.

**Thứ tự cắt nếu vượt ngân sách.** (1) Bỏ nhân bản PhoBERT và RoBERTa. (2) Bỏ mức dữ liệu 50%. (3) Giảm seed C2 và C4 từ 5 xuống 3 (tiết kiệm ~48 giờ). (4) Giảm mẫu tính cộng từ 40 xuống 20. (5) Thuê GPU ngoài. Phần không cắt: pilot, C1, C3, đối chứng oracle.

## 12. Kịch bản kết quả và cách diễn giải

Mỗi kết quả có một cách diễn giải được khai báo trước, và không kết quả nào được phép diễn giải mạnh hơn mức power cho phép.

| Kết quả                             | Diễn giải được phép                                                                                                                                                       | Không được phép nói                                 |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| H1/H2 đạt, H4 đạt, H5 đạt, H7 đạt   | Capacity need là thuộc tính ổn định, proxy dự đoán được, và phân bổ theo đó có lợi. Xác nhận giả định nền tảng; ít đột phá hơn nhưng là nền cho các bảng khuyến nghị rank | "Lần đầu tiên chứng minh"                           |
| H1/H2 đạt, H4 không đạt             | Capacity need đo được nhưng phụ thuộc task, ngôn ngữ, dữ liệu hoặc prior; không có bảng rank tĩnh                                                                         | "Capacity need không tồn tại"                       |
| H1/H2 đạt, H5 không đạt             | Profile đo được nhưng các proxy hiện tại không dự đoán được; điểm số của các phương pháp thích ứng không thể giải thích bằng capacity need đo được                        | "Các phương pháp thích ứng không hoạt động"         |
| H7 oracle không vượt đều (TOST đạt) | Biết capacity need chính xác cũng không cải thiện hiệu năng; ngân sách quan trọng hơn cách phân bổ trong phạm vi này                                                      | Tổng quát sang mô hình lớn hơn                      |
| H6 đạt                              | Mất cân bằng lớp làm nhiễu proxy dựa trên gradient; cân bằng lớp là bước cần thiết                                                                                        | "Fisher hỏng" khi dữ liệu cân bằng                  |
| G1 không đạt (ρ \< 0,2)             | Profile không đủ độ phân giải ở mức một lớp trong phạm vi thiết kế này. Chuyển sang đơn vị thô và báo cáo với power                                                       | "Phương pháp của AdaLoRA, IGU-LoRA, BaRA bị bác bỏ" |

Bốn phân loại của bản 3 (tương quan cao hay thấp × Dynamic vượt hay không) vẫn đúng và được giữ ở mức phân tích, nhưng nay được đặt sau câu hỏi đo được trước.

## 13. Hạn chế, rủi ro, tách Paper B và đầu ra

Rủi ro lớn nhất là profile không vượt được nhiễu ở mức một lớp; đã có cổng G1 để phát hiện sớm và đơn vị thô để chuyển hướng.

**Hạn chế.**

- Chỉ encoder-based, chỉ phân loại văn bản, hai ngôn ngữ (Việt và Anh). Kết quả không tổng quát tới mô hình decoder hay ngôn ngữ khác.

- Capacity need có điều kiện theo ngữ cảnh (các lớp khác ở rank 8). H3 kiểm tra điều này nhưng chỉ trên một cell.

- Chỉ 12 lớp nên kiểm định xếp hạng từng cell có power thấp; gộp qua cell giảm bớt nhưng không loại bỏ.

- Nếu chênh lệch Macro-F1 giữa các lớp chỉ cỡ vài phần trăm hoặc nhỏ hơn, mọi thứ có thể nằm trong nhiễu; cổng G1 xử lý tình huống này.

- Mất cân bằng lớp được thao túng bằng cấp số nhân ở N nhỏ, nên không đại diện cho mọi dạng mất cân bằng tự nhiên.

**Rủi ro về tính mới.** Rank allocation đông đúc (AdaLoRA, IGU-LoRA, BaRA, StatLoRA, RSLoRA). Giá trị của đề tài nằm ở phép đo và kiểm toán, không phải thuật toán mới, và bài viết bám vào phần này. Rà soát tài liệu phải cập nhật ngay trước khi nộp vì lĩnh vực đang ra bài mới theo tháng.

**Tách thành hai bài.**

- **Paper A (bản đề xuất này).** RQ1 đến RQ3: định nghĩa, nhiễu, ổn định, chuyển giao, proxy, mất cân bằng, downstream.

- **Paper B (đề xuất riêng, hoặc phụ lục).** Calibration (PEFT có hiệu chuẩn kém hơn FFT không; Temperature Scaling trên calibration split riêng) và Pareto F1–NLL–VRAM–latency. Dùng lại logit và cấu hình đã chạy ở Paper A; calibration split được lưu sẵn ngay từ đầu.

**Đầu ra dự kiến.**

- Bản thảo Paper A gửi tạp chí Scopus Q1/Q2 (mục tiêu, chưa cam kết).

- Mã nguồn mở, kế hoạch thống kê đăng ký trước, và bản xuất F1 theo từng mẫu.

- Công cụ kiểm toán rank: nhận một mẫu phân bổ hoặc một proxy, trả về độ tin cậy và τb so với profile đo được.

- Profile L_m theo layer và theo cell, kèm bảng khuyến nghị rank chỉ khi H4 đạt.

## 14. Tài liệu tham khảo

Các mục \[1\]–\[9\] và \[12\]–\[20\] kế thừa từ bản 3. Mục \[10\], \[11\], \[21\], \[22\] đã được xác nhận tồn tại qua trang bài báo, nhưng danh sách tác giả của \[11\], \[21\], \[22\] chưa được đối chiếu và \[21\], \[22\] mới được đọc phần tóm tắt; cần đọc toàn văn và điền đủ tác giả trước khi nộp.

- \[1\] Nguyen, D.Q., Nguyen, A.T.: PhoBERT: pre-trained language models for Vietnamese. In: Findings of the ACL: EMNLP 2020, pp. 1037–1042 (2020)

- \[2\] Nguyen, N.Q., Phan, T.D., Nguyen, D.V., Nguyen, K.V.: ViSoBERT: a pre-trained language model for Vietnamese social media text processing. In: EMNLP 2023, pp. 5191–5207 (2023)

- \[3\] Conneau, A., et al.: Unsupervised cross-lingual representation learning at scale. In: ACL 2020, pp. 8440–8451 (2020)

- \[4\] Hu, E.J., et al.: LoRA: low-rank adaptation of large language models. In: ICLR 2022 (2022)

- \[5\] Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L.: QLoRA: efficient finetuning of quantized LLMs. In: NeurIPS 36 (2023)

- \[6\] Aghajanyan, A., Zettlemoyer, L., Gupta, S.: Intrinsic dimensionality explains the effectiveness of language model fine-tuning. In: ACL-IJCNLP 2021, pp. 7319–7328 (2021)

- \[7\] Frankle, J., Carbin, M.: The lottery ticket hypothesis: finding sparse, trainable neural networks. In: ICLR 2019 (2019)

- \[8\] Roy, O., Vetterli, M.: The effective rank: a measure of effective dimensionality. In: EUSIPCO 2007, pp. 606–610 (2007)

- \[9\] Zhang, Q., et al.: Adaptive budget allocation for parameter-efficient fine-tuning. In: ICLR 2023 (2023)

- \[10\] Cui, X., et al.: IGU-LoRA: adaptive rank allocation via integrated gradients and uncertainty-aware scoring. In: ICLR 2026; arXiv:2603.13792 (2026). [Link](https://arxiv.org/abs/2603.13792)

- \[11\] Duan, Z., et al.: BaRA: Bayesian adaptive rank allocation for parameter-efficient fine-tuning. arXiv:2606.29184 (2026). [Link](https://arxiv.org/pdf/2606.29184)

- \[12\] Nguyen, K.V., et al.: UIT-VSFC: Vietnamese students' feedback corpus for sentiment analysis. In: KSE 2018, pp. 19–24. IEEE (2018)

- \[13\] Ho, V.A., et al.: Emotion recognition for Vietnamese social media text. In: PACLING 2019. CCIS, vol. 1215, pp. 319–333. Springer (2020)

- \[14\] Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q.: On calibration of modern neural networks. In: ICML 2017, pp. 1321–1330 (2017)

- \[15\] Kornblith, S., Norouzi, M., Lee, H., Hinton, G.: Similarity of neural network representations revisited. In: ICML 2019, PMLR 97, pp. 3519–3529 (2019)

- \[16\] Wilcoxon, F.: Individual comparisons by ranking methods. Biom. Bull. 1(6), 80–83 (1945)

- \[17\] Holm, S.: A simple sequentially rejective multiple test procedure. Scand. J. Stat. 6(2), 65–70 (1979)

- \[18\] Lakens, D.: Equivalence tests: a practical primer. Soc. Psychol. Personal. Sci. 8(4), 355–362 (2017)

- \[19\] Kendall, M.G.: The treatment of ties in ranking problems. Biometrika 33(3), 239–251 (1945)

- \[20\] El-Yaniv, R., Wiener, Y.: On the foundations of noise-free selective classification. JMLR 11, 1605–1641 (2010)

- \[21\] Statistical inference for rank allocation in low-rank adaptation (StatLoRA). arXiv:2607.20205 (2026). [Link](https://arxiv.org/abs/2607.20205)

- \[22\] RSLoRA: training-free rank allocation for LoRA via representational sensitivity probing. arXiv:2607.09757 (2026). [Link](https://www.alphaxiv.org/abs/2607.09757v1)

- \[23\] Bates, D., Mächler, M., Bolker, B., Walker, S.: Fitting linear mixed-effects models using lme4. J. Stat. Softw. 67(1), 1–48 (2015)

- \[24\] Spearman, C.: The proof and measurement of association between two things. Am. J. Psychol. 15(1), 72–101 (1904)

# Hạn chế



Tổng hợp từ `analysis_plan_v2.0.md`, `pivot_v2.md`, `exploratory_results.md`.

Cập nhật khi phát hiện thêm. Dùng làm nguồn trích dẫn khi viết bài.



---



## 1. Phạm vi



- Một mô hình (XLM-R-base), một ngôn ngữ (tiếng Việt), một task chính (UIT-VSFC sentiment 3 lớp).

- Split cố định — nhiễu do chọn tập huấn luyện không nằm trong phương sai seed.

- Encoder, phân loại văn bản. Không kết luận cho decoder hoặc LLM lớn.



## 2. Dữ liệu



- UIT-VSFC: lớp neutral ~4% train (522 mẫu), ~67 mẫu test.

- Macro-F1 bị chi phối bởi lớp neutral. δ = 0,01 trên Macro-F1 ≈ 1–2 mẫu neutral.

- Near-duplicate (TF-IDF char_wb 3-5 gram, cosine ≥ 0.95):

&#x20; - calibration: 5.19%

&#x20; - test: 3.81%

- Chưa kiểm tra giả định "leak ảnh hưởng mọi rank như nhau" (rank cao có thể ghi nhớ leak nhiều hơn).



## 3. Thiết kế



- LR tune một lần ở 6 epoch, tái dùng cho 3 epoch (không tune lại).

- LR chốt ở lưới {1e-4, 2e-4, 5e-4, 1e-3} — tối ưu có thể ngoài lưới, đặc biệt rank 4 (F1 tăng đơn điệu theo LR).

- α = 2r (α/r = 2 cố định). Độ nhạy α (fixed:16, rsLoRA) ở mức C, chưa chạy.

- Full fine-tuning dùng LR 2e-5 (khác LoRA). LR 2e-4 làm full FT collapse về majority class.



## 4. Kỹ thuật



- **Precision:** fp16 thay vì bf16. T4 = SM 7.5 không có bf16 gốc. `torch.cuda.is_bf16_supported()` trả True nhầm do bug PyTorch. Benchmark: bf16 mô phỏng 21,9s vs fp16 6,7s (3,27×).

- **3 cấu hình collapse ở tune seed 400:** rank 16 + LR 1e-3, rank 32 + LR 5e-4, rank 32 + LR 1e-3. F1 = 0.2207, NLL ≈ 0.837 (khớp entropy lớp train).

- **Reference baselines** dùng seed dải riêng (600–607 cho head, 600–604 cho full FT), không so trực tiếp với R1.



## 5. Nhiễu và power



- SD dư 0,0081 đo từ Stage 1c (calibration, 6 epoch, module-level, bf16).

- Chưa đo trực tiếp trên test + fp16 + LR per rank + rank toàn cục.

- Power 0,89 sau Holm (N = 16, δ = 0,01). Nếu SD thực tế lớn hơn 25% (0,0101), power còn ~0,60.

- 5 lượt full FT đầu dùng LR 2e-4 bị collapse — ghi vào sai lệch, không dùng.



## 6. Đăng ký trước



- Tag git nội bộ (`analysis-plan-v2.0`) + OSF Preregistration (không phải "Registered Report" hay "AsPredicted").

- OSF Preregistration không có peer review trước khi submit, không ràng buộc tạp chí.

- Viết sau khi đã thấy dữ liệu khám phá (module-level). Không phải blind pre-registration.

- Tuy nhiên: seed ranges tách biệt rõ (100–114 exploration, 200–215 confirmation), tag commit hash, không có thay đổi sau tag v2.0.



## 7. Kết luận — điều không được nói



- Không nói "rank không quan trọng" nói chung.

- Không nói "capacity need không tồn tại".

- Không nói "AdaLoRA/IGU-LoRA/BaRA/StatLoRA tối ưu trên nhiễu".

- Không nói "kiểm định đầu tiên".

- Mọi so sánh ghép cặp theo seed, trên cùng tập test. Không khái quát ra ngoài phạm vi đã thử.



## 8. Ghi chú thêm từ peer review nội bộ (2026-10-16)



Các điểm sau đã được ghi nhận và xử lý ở mức thiết kế, nhưng cần nhắc lại trong bài:



- **Căn cứ đổi hướng:** các phân tích khám phá của hướng cũ (Stage 1c) đổi rank một lớp/module, không đo rank toàn cục. Con số gộp F1 −0,0013 không dùng để justify H1 (đó là giả thuyết kiểm chứng được, không phải niềm tin).

- **F1 neutral chiếm ưu thế:** Macro-F1 trên VSFC bị chi phối bởi lớp neutral (~4% train, 67 mẫu test). Báo cáo kèm F1 từng lớp và bootstrap trên test.

- **LR tune trên F1, không trên NLL:** H2 có thể bị confound với LR khác nhau giữa rank. Đã xử lý bằng nhánh LR cố định 2e-4 làm robustness check.

- **Tune LR 1 seed (đến 2026-10-16):** đang chạy thêm seed 401, 402 để chốt LR per rank. Kết luận "LR tối ưu khác nhau giữa rank" chỉ nên nói khi có 3 seed.

- **α = 2r làm rank dính với bước học:** độ nhạy α cần chạy sớm (mức C) để tách ảnh hưởng rank thuần túy khỏi scaling.


## 9. Hạn chế mới phát hiện từ tune seed 400 (2026-10-17)

### 9.1. Lưới LR chạm biên

Kết quả tune seed 400 (thăm dò, 1 seed) cho thấy LR tối ưu nằm ở **biên lưới** cho 4/5 rank:

| Rank | Tối ưu | Vị trí |
|---|---|---|
| 1 | 2e-4 | trong lưới |
| 4 | 1e-3 | **biên trên** |
| 8 | 1e-4 | **biên dưới** |
| 16 | 1e-4 (sau quy tắc hòa) | **biên dưới** |
| 32 | 1e-4 | **biên dưới** |

Hệ quả: nếu không mở rộng lưới, kết luận "rank toàn cục không giúp ích" có thể sai vì lưới chưa bao phủ tối ưu thật của rank cao.

**Xử lý đã đăng ký:** quy tắc mở rộng lưới một bước (5e-5 ở biên dưới, 2e-3 ở biên trên), ghi vào `analysis_plan_v2.0.md` mục 9 ngày 2026-10-17, **trước khi đọc kết quả seed 401–402 và trước R1**. Giới hạn: chỉ mở rộng **một bước**, không lặp.

### 9.2. Ba lượt collapse ở seed 400 — tương tác rank × LR

3/20 lượt tune seed 400 collapse (Macro-F1 calibration ≈ 0,2207, khớp dự đoán lớp đa số):

| Rank | LR | F1 (cal) | NLL (cal) |
|---|---|---|---|
| 16 | 1e-3 | 0,2207 | ~0,837 |
| 32 | 5e-4 | 0,2207 | ~0,837 |
| 32 | 1e-3 | 0,2207 | ~0,837 |

Đây là **bằng chứng cho tương tác rank × LR**: với α = 2r, rank cao có biên độ ΔW lớn hơn, không chịu được LR cao. Tương tác này không đăng ký trước cho tune seed 400, nhưng được báo cáo ở mục 7 (Phân tích thứ cấp) của kế hoạch v2.0 với nhãn "khám phá".

**Xử lý:** 3 lượt collapse **giữ** trong bảng tuning và trong trung bình chọn LR (không loại bỏ). Định nghĩa "sập" (cal F1 < 0,25) đăng ký vào mục 9 ngày 2026-10-17, trước R1. Tỷ lệ sập theo (rank, LR) sẽ báo cáo ở phụ lục.

### 9.3. Định nghĩa "sập" ra đời sau khi chạy seed 400

Định nghĩa "sập" (< 0,25) được viết khi đã **thấy** 3 lượt collapse ở seed 400 — không phải quy tắc đăng ký từ trước. Đây là điểm yếu của hướng khám phá, không phải của R1.

**Xử lý:** định nghĩa được ghi vào mục 9 của `analysis_plan_v2.0.md` **trước khi chạy R1**, kèm quy tắc xử lý ở R1 (giữ hay bỏ). Seed tune (400–402) là dữ liệu thăm dò, không phải dữ liệu xác nhận; seed R1 (200–215) chưa chạy.

### 9.4. `--lr_map` patch sau khi biết seed 400

`run_pilot.py` thêm `--lr_map` (LR per rank) ngày 2026-10-17, **sau** khi thấy seed 400 có LR tối ưu khác nhau giữa các rank. Đây là hạ tầng kỹ thuật, không phải thay đổi thiết kế — nhưng cần ghi để minh bạch.

**Xử lý:** patch ghi vào CHANGELOG (Cycle 3b, commit `4148c0b`). Kiểm tra: `run_pilot.py` giữ nguyên logic huấn luyện giữa `6ef174b` (chạy seed 400) và các commit sau — chỉ thêm nhánh `if lr_map is not None`, không ảnh hưởng khi `lr_map=None` (backward compat).

### 9.5. Suy biến ma trận ở nhánh robustness của H3 (R1)

Trong phân tích R1, nhánh robustness của H3 (sử dụng LR cố định 2e-4) gặp hiện tượng `SingularMatrixWarning` (ma trận thiết kế bị rank-deficient), dẫn đến kết quả `F=inf, p=nan`.

**Nguyên nhân:** R1 là kết quả của nhánh LR per rank. Do đó, trong file dữ liệu `global_merged.csv`, chỉ có rank 16 là được chạy ở mức LR 2e-4 (theo tuning_decision.json). Các rank khác (1, 4, 8, 32) được chạy ở LR 1e-3, 5e-4, 1e-4, 5e-5. Vì vậy, nhánh LR cố định 2e-4 chỉ có dữ liệu của rank 16, dẫn đến bậc tự do (df) = 0 và không thể ước lượng được hệ số. 

**Xử lý:** 
- Đây là hiện tượng **thiếu dữ liệu (missing data)** của nhánh robustness, không phải "sập mô hình hàng loạt". 
- Nhánh chính của H3 (LR per rank) hoàn toàn vững chắc: F = 9.613, p thô = 0.0023, p_holm = 0.0070.
- Để khắc phục triệt để trong tương lai, nhánh robustness đủ 5 rank ở LR 2e-4 sẽ được chạy bổ sung (V1 - xem kế hoạch tiếp theo).
### 9.6. Độ nhạy của FP16 ở ngưỡng biên

Thí nghiệm V5 (FP32, 1 seed) cho thấy hiện tượng sập ở LR=1e-3 là bản chất của tối ưu hóa. Ở LR=5e-4, Rank 32 ổn định ở FP32. Tuy nhiên, do V5 mới chạy 1 seed (200) so với FP16 (seed 400-402), chưa thể kết luận chắc chắn về tác động của FP16. Cần thêm dữ liệu ghép cặp theo seed (V5b) để xác nhận. Kết quả chính (R1) dùng FP16 với LR an toàn (5e-5) nên không bị ảnh hưởng.

### 9.7. H3 không bền vững với cách chọn LR (Thí nghiệm V1)

Kết quả R1 cho thấy H3 có ý nghĩa (p_holm = 0.007) với tương tác rank × epoch. Tuy nhiên, thí nghiệm hậu kiểm V1 (16 lượt, 4 ranks, 3 epochs, LR cố định 2e-4) cho thấy khi LR được cố định, F1 của Rank 8 và 32 hồi phục hoàn toàn (từ 0.805/0.799 lên 0.827/0.842). Điều này chứng tỏ tương tác trong R1 phần lớn là do việc dùng LR của 6 epoch (đã tune) cho 3 epoch. Do đó, câu kết luận đúng là: **H3 không bền vững**. Tương tác rank × epoch phụ thuộc vào cách chọn LR: đảo dấu giữa nhánh LR theo rank (+0.017) và nhánh LR cố định (-0.033).

## EXT-1: hạn chế bổ sung (2026-10-26)
- So sánh PhoBERT và XLM-R đồng thời thay đổi tokenizer, tách từ, từ vựng, dữ liệu tiền huấn luyện và số tham số; tương tác mô hình × rank (E1) không quy được cho nguyên nhân kiến trúc cụ thể.
- Hướng hiệu ứng của PhoBERT đã được thấy ở B1/V4 trước khi đăng ký EXT-1; EXT-1 là kiểm tra lặp lại và mở rộng trên lượt chạy mới, không phải phát hiện độc lập.
- Chỉ một tác vụ (UIT-VSFC sentiment) có đủ quy trình; B2 (AG News) dùng LR cố định.
- H3 (rank × epoch) chỉ được kiểm cho XLM-R nếu không chạy mức P3 (PhoBERT 3 epoch).
- Hiệu chuẩn (ECE, temperature scaling) và ΔW cho PhoBERT chỉ có từ lần chạy B1X (logits và ΔW của B1 cũ không được lưu trong repo).

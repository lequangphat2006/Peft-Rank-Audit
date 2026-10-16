# Kế hoạch phân tích (đăng ký trước) — Global rank sweep, v2.0

Trạng thái: BẢN NHÁP. Tag `analysis-plan-v2.0-draft` được gắn khi commit bản này
(đủ để bắt đầu tune). Tag `analysis-plan-v2.0` được gắn sau khi TS. Huyên xem
(bắt buộc trước khi chạy bất kỳ lượt R1 nào).
Mọi thay đổi sau tag v2.0 ghi vào mục 9.

---

## 0. Tóm tắt một đoạn

Câu hỏi: khi LR được tune công bằng theo từng rank, α = 2r, trên XLM-R-base với
UIT-VSFC, rank LoRA toàn cục ảnh hưởng thế nào đến Macro-F1 và NLL trên test,
và hiệu ứng đó có phụ thuộc ngân sách huấn luyện (3 so với 6 epoch) không?
H1–H3 kiểm định bằng TOST (δ = 0,01), hệ số NLL một phía, và F-test tương tác,
với Holm m = 3, α = 0,05. N = 16 seed (200–215).

---

## 1. Câu hỏi và phạm vi

**Câu hỏi chính.** Trong thiết lập XLM-R-base + LoRA + UIT-VSFC sentiment, LR tune
theo từng rank, α = 2r:

- (a) Tăng rank toàn cục (cùng rank cho 12 lớp × 6 module) có cải thiện Macro-F1 không?
- (b) Rank ảnh hưởng thế nào đến NLL (trước và sau temperature scaling)?
- (c) Hiệu ứng có thay đổi giữa 3 và 6 epoch không?

**Câu hỏi phụ (mức L):** 10 epoch và 20%/50% data có làm đổi kết luận không?
F1/NLL trên cell B1, B2 có nhất quán không?

**Phạm vi.** Encoder, phân loại văn bản. Không kết luận cho decoder, LLM lớn,
hoặc ngôn ngữ ngoài tập đã thử.

---

## 2. Thiết kế

### 2.1 Cell R1 (chính)

| Thành phần | Giá trị |
|---|---|
| Task | UIT-VSFC sentiment (3 lớp) |
| Model | FacebookAI/xlm-roberta-base |
| Split | Train 12.780 / Cal 1.598 / Test 1.599 (cố định) |
| Rank | {1, 4, 8, 16, 32} |
| α | 2r (`alpha_pattern`) |
| Target modules | 6/layer, 72 module |
| Ngân sách | 3, 6 epoch (chính); 10, 20%, 50% (mức L, mục 4.2) |
| Seed R1 | 200–215 (16) |
| Seed tuning | 400–402 (3) |
| Seed thử σ | 500–502 (3, mỗi cell B) |
| Siêu tham số cố định | batch 32, max_len 128, warmup 10% + cosine, wd 0.01, clip 1.0, LoRA dropout 0.1 |
| Loss | Cross-entropy **không trọng số** |
| Precision | **Chốt sau khi kiểm tra T4**; ghi vào CSV; không đổi giữa các lượt của cùng so sánh |
| Đánh giá | **Test** là chỉ số chính. Calibration chỉ dùng để chọn LR (tune) và fit nhiệt độ |

### 2.2 Cell B (mức L: B1, B2; mức C: B3–B5)

| Cell | Task | Model | Rank | Seed | Ghi chú |
|---|---|---|---|---|---|
| B1 | VSFC sentiment | ViSoBERT hoặc PhoBERT | {1, 4, 16, 32} | 300–307 | PhoBERT cần tách từ |
| B2 | UIT-VSMEC | XLM-R-base | {1, 4, 16, 32} | 300–307 | VSMEC nhỏ hơn (~45% VSFC) |
| B3 | VSMEC | ViSoBERT/PhoBERT | {1, 4, 16, 32} | 300–307 | Mức C |
| B4 | Task Anh | XLM-R-base | {1, 4, 16, 32} | 300–307 | AG News hoặc SST-2 (~12k) |
| B5 | Task Anh | RoBERTa-base | {1, 4, 16, 32} | 300–307 | Mức C |

### 2.3 Đơn vị lượt

Mỗi `run_id` = hash cell × rank × LR × alpha_mode × ngân sách × seed.
Không ghi đè. `run_pilot.py` resume theo `run_id`.

---

## 3. Chỉ số

**Chính (test):**
- Macro-F1
- NLL không trọng số (CE trung bình trên test)

**Phụ:**
- Accuracy, F1 từng lớp (đặc biệt lớp neutral ~4%)
- NLL và ECE sau temperature scaling (T fit trên calibration của chính lượt đó)
- Train loss theo epoch
- Thống kê ΔW: effective rank kiểu entropy, stable rank, chuẩn Frobenius

**CI:** 95% cho báo cáo chính; **90% cho TOST** (tương đương hai kiểm định một phía ở 0,05).

---

## 4. Giả thuyết xác nhận (họ H1–H3, Holm m = 3, α = 0,05)

Mô hình: `metric ~ C(seed) + log2(rank)`. Hiệu ứng tổng 1→32 = hệ số × log2(32/1) = hệ số × 5.

| Mã | Giả thuyết | Kiểm định | Quy tắc |
|---|---|---|---|
| **H1** | Hiệu ứng tổng rank 1→32 lên Macro-F1 tương đương 0 ở 6 epoch, **LR theo rank** | TOST, δ = 0,01, CI 90% | Tương đương nếu p_holm < 0,05 |
| **H2** | NLL tăng theo rank ở 6 epoch, **LR theo rank** | Hệ số log2(rank) > 0, một phía | Xác nhận nếu p_holm < 0,05 |
| **H3** | Độ dốc rank của Macro-F1 khác nhau giữa 3 và 6 epoch | F-test tương tác `lr:C(epochs)` (cùng seed) | Có ý nghĩa nếu p_holm < 0,05 |

**Ba điều kiện bền (bắt buộc báo cáo kèm):**

1. **H1, H2 giữ nguyên kết luận khi dùng LR tốt nhì** (LR cao thứ hai theo calibration, khác giá trị với LR đã chọn).
2. **H3 nhánh chính là nhánh LR theo rank.** Nhánh LR cố định 2e-4 là **kiểm tra độ bền**, không phải kiểm định thứ hai. H3 chỉ được coi là bền khi hai nhánh cùng dấu và ước lượng tương tác của nhánh cố định nằm trong CI 95% của nhánh chính.
3. **Ghi rõ:** LR chỉ tune ở 6 epoch, không tune lại cho 3 epoch.

**Lý do chọn δ = 0,01:**
- 1 điểm Macro-F1 — ngưỡng có ý nghĩa thực tế khi so sánh phương pháp PEFT.
- Power với σ_F1 = 0,0081: δ = 0,005 không khả thi (power tối đa 0,19 ở N = 16); δ = 0,01 đạt 0,95.
- Báo cáo thêm độ nhạy ở δ ∈ {0,005; 0,0075; 0,01} với CI 90%, gắn nhãn thăm dò.

**δ tương đối (mục 4.6 v4.1):** δ giữ tuyệt đối 0,01 ở mọi cell. Báo cáo thêm bảng độ nhạy
δ_cell = 0,01 × F1_cell / F1_R1 (F1 tại rank 8, 6 epoch). Nêu cell nào đổi kết luận.

---

## 5. Quy tắc chọn LR (ghi trước, dùng ở Giai đoạn 1)

- Với mỗi rank, chạy lưới LR {1e-4, 2e-4, 5e-4, 1e-3} × seed {400, 401, 402}, 6 epoch, **trên calibration**.
- Chọn LR có Macro-F1 trung bình calibration (trung bình 3 seed) cao nhất.
- **Ngưỡng hòa: 0,005.** Hai LR được coi là hòa nếu chênh lệch trung bình < 0,005. Khi hòa, chọn LR **nhỏ hơn**.
- **LR tốt nhì:** LR khác giá trị với LR đã chọn, có Macro-F1 calibration cao thứ hai, kể cả khi hai LR hòa.
- Báo cáo toàn bộ bảng tuning ở phụ lục.
- **Không chỉnh ngưỡng sau khi thấy dữ liệu.** Nếu ngưỡng tỏ ra không phù hợp, giữ nguyên và ghi vào mục 9.

Ngưỡng 0,005 ≈ một sai số chuẩn theo seed với SD dư 0,0081 và 3 seed (0,0081/√3 ≈ 0,0047).

**Cổng A (sau tune):**
- (a) Nếu LR tối ưu dịch theo rank → mọi giai đoạn sau dùng LR theo rank.
- (b) Nếu ở LR đã tune, rank 1→32 đã lệch ≥ δ trên calibration → vẫn chạy tiếp, nhưng câu chuyện đổi thành "khi nào rank quan trọng".
- (c) Nếu phẳng → tiếp tục đúng thiết kế.

---

## 6. Cỡ mẫu và quy tắc dừng

**N = 16 seed (200–215)** cho R1 chính. Chia 2 đợt:
- Đợt 1: seed 200–207 (8 seed)
- Đợt 2: seed 208–215 (8 seed)

**Quy tắc dừng:**
- Không phân tích H1–H3 giữa hai đợt.
- Sau đợt 1, chỉ chạy `power_global_sweep.py` để kiểm tra residual SD.
- Nếu residual SD > 0,009 → mở rộng lên 20 seed (216–219), ghi mục 9.
- Nếu sau N seed CI rộng hơn δ → báo cáo đúng "chưa kết luận được".

---

## 7. Phân tích thứ cấp (đã chỉ định trước, báo cáo kèm CI, không khẳng định bằng p-value)

- Hiệu ứng từng ngân sách (3 và 6 epoch riêng)
- NLL/ECE sau temperature scaling
- F1 từng lớp
- Đường hội tụ (train loss theo epoch)
- ΔW: effective rank, stable rank, chuẩn Frobenius so với rank danh nghĩa
- Mốc trần (full FT) và sàn (head-only)
- Độ nhạy α (2r vs fixed:16 vs rsLoRA) — mức C
- Nhánh LR cố định và LR tốt nhì
- 10 epoch và 20%/50% data (mức L)
- OOD (mức C, chỉ khi có tập đạt tiêu chí ở Giai đoạn 6)
- Hiệu ứng từng cell B

**Phân tích gộp qua cell (thăm dò):** forest plot, ước lượng gộp CI (Hartung–Knapp), gắn nhãn mô tả. Không diễn giải τ như kết quả.

---

## 8. Quy tắc thực hành (ghi trước)

**Dạng quy tắc:** rank nhỏ nhất mà Macro-F1 nằm trong δ = 0,01 của rank tốt nhất **và** NLL không tệ hơn δ_NLL = 0,01.

**Kiểm lại:** leave-one-cell-out khi có ≥ 5 cell (mức C). Ở mức L chỉ trình bày như gợi ý mô tả.

---

## 9. Sai lệch so với kế hoạch (điền khi xảy ra)

| Ngày | Thay đổi | Lý do | Ảnh hưởng đến kết luận |
|---|---|---|---|
| 2026-10-16 | Precision Phase 1: chốt **fp16** (thay bf16) | T4 = SM 7.5, không có bf16 gốc. `torch.cuda.is_bf16_supported()` báo True nhầm (chỉ kiểm CUDA version). Benchmark thật: bf16 mô phỏng 21,9s/lượt, fp16 6,7s/lượt → **3,27× nhanh hơn**. Phase 1 dùng seed mới nên không so với dữ liệu cũ | Không ảnh hưởng kết luận; mọi lượt Phase 1 dùng fp16 nhất quán; `resolve_precision` tự chọn fp16 khi compute capability < 8.0 |
| 2026-10-16 | VSMEC: dự kiến tải từ github.com/uitnlp/UIT-VSMEC, upload Kaggle Dataset riêng | `uitnlp/UIT-VSMEC` không có trên HF Hub (401); `glue/sst2` cũng vậy | Nếu không kịp trước 11-04: fallback `SetFit/ag_news` cho B2-B4, ghi limitations |

---

## 10. Kết quả được phép diễn giải trước

| Kết quả | Được phép nói | Không được nói |
|---|---|---|
| H1 đạt, H2 đạt | Trong các thiết lập đã thử, rank không cải thiện phân biệt (trong ±δ) và làm NLL tệ nhẹ | "Rank không quan trọng" nói chung |
| Cổng A: F1 tăng theo rank ở LR đã tune | Kết luận về rank phụ thuộc cách tune; so sánh rank dưới một LR chung có thể thiên vị | "Rank thấp luôn đủ" |
| NLL biến mất sau temperature scaling | Phần NLL do overconfidence, không phải dung lượng | Cơ chế khi chưa kiểm ΔW và hội tụ |
| H3 đạt ở cả hai nhánh | Độ nhạy rank phụ thuộc ngân sách | "Rank thấp chỉ chậm hơn" (khi chưa có 10 epoch) |
| H3 chỉ đạt một nhánh | Tương tác chưa bền với cách chọn LR | Kết luận về ngân sách |
| Dị biệt lớn giữa cell | Hiệu ứng phụ thuộc task hoặc mô hình | "Một quy tắc chung" |
| CI rộng hơn δ | Chưa đủ bằng chứng | "Không có hiệu ứng" |

---

## 11. Quy tắc vận hành

- Lượt NaN vẫn giữ, đếm, báo cáo. Không thay seed. Có phân tích độ nhạy khi loại.
- Không thêm seed ngoài quy tắc dừng.
- Mọi thay đổi ghi mục 9 kèm lý do và ngày.
- δ chính giữ tuyệt đối 0,01 ở mọi cell; σ từng cell dùng để tính power, không đổi δ.

---

## 12. Sổ phân tích đã thực hiện trên dữ liệu cũ (khám phá, không dùng để xác nhận)

Giữ nguyên từ `analysis_plan.md` v1 (mục 8) — xem file cũ.
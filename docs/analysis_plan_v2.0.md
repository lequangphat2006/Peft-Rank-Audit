\# Kế hoạch phân tích (đăng ký trước) — Global rank sweep, v2.0



Trạng thái: \*\*BẢN CHÍNH THỨC từ 2026-10-17\*\*, tag `analysis-plan-v2.0`.

Tag được gắn bởi tác giả với toàn quyền quyết định từ TS. Huyên (xem `docs/decisions.md`).

Mọi thay đổi sau tag ghi vào mục 9 (Sai lệch).



Bối cảnh: kế hoạch này thuộc hướng đề tài rank LoRA toàn cục; đề tài ban đầu về capacity need của từng lớp đã bị thay thế, xem `docs/pivot\_v2.md`. Nhãn H1–H3 dưới đây \*\*không\*\* trùng nghĩa với H1–H7 của bản đề xuất 4.



\---



\## 0. Tóm tắt một đoạn



Câu hỏi: khi LR được tune công bằng theo từng rank, α = 2r, trên XLM-R-base với

UIT-VSFC, rank LoRA toàn cục ảnh hưởng thế nào đến Macro-F1 và NLL trên test,

và hiệu ứng đó có phụ thuộc ngân sách huấn luyện (3 so với 6 epoch) không?

H1–H3 kiểm định bằng TOST (δ = 0,01), hệ số NLL một phía, và F-test tương tác,

với Holm m = 3, α = 0,05. N = 16 seed (200–215).



\---



\## 1. Câu hỏi và phạm vi



\*\*Câu hỏi chính.\*\* Trong thiết lập XLM-R-base + LoRA + UIT-VSFC sentiment, LR tune

theo từng rank, α = 2r:



\- (a) Tăng rank toàn cục (cùng rank cho 12 lớp × 6 module) có cải thiện Macro-F1 không?

\- (b) Rank ảnh hưởng thế nào đến NLL (trước và sau temperature scaling)?

\- (c) Hiệu ứng có thay đổi giữa 3 và 6 epoch không?



\*\*Câu hỏi phụ (mức L):\*\* 10 epoch và 20%/50% data có làm đổi kết luận không?

F1/NLL trên cell B1, B2 có nhất quán không?



\*\*Phạm vi.\*\* Encoder, phân loại văn bản. Không kết luận cho decoder, LLM lớn,

hoặc ngôn ngữ ngoài tập đã thử.



\---



\## 2. Thiết kế



\### 2.1 Cell R1 (chính)



| Thành phần | Giá trị |

|---|---|

| Task | UIT-VSFC sentiment (3 lớp) |

| Model | FacebookAI/xlm-roberta-base |

| Split | Train 12.780 / Cal 1.598 / Test 1.599 (cố định) |

| Rank | {1, 4, 8, 16, 32} |

| α | 2r (`alpha\_pattern`) |

| Target modules | 6/layer, 72 module |

| Ngân sách | 3, 6 epoch (chính); 10, 20%, 50% (mức L, mục 4.2) |

| Seed R1 | 200–215 (16) |

| Seed tuning | 400–402 (3) |

| Seed reference | 600–607 (head + full FT) |

| Seed thử σ | 500–502 (mỗi cell B) |

| Siêu tham số cố định | batch 32, max\_len 128, warmup 10% + cosine, wd 0.01, clip 1.0, LoRA dropout 0.1 |

| Loss | Cross-entropy \*\*không trọng số\*\* |

| \*\*Precision\*\* | \*\*fp16\*\* (benchmark 2026-10-16: bf16 21,9s vs fp16 6,7s = 3,27× trên T4 SM 7.5) |

| Đánh giá | \*\*Test\*\* là chỉ số chính. Calibration chỉ dùng để chọn LR (tune) và fit nhiệt độ |



\*\*LR theo rank:\*\* chốt từ tune 3 seed (400–402) trên calibration. Quy tắc chọn ở mục 5.

\*\*LR full FT:\*\* 2e-5 (khác LoRA; LR 2e-4 làm full FT collapse, xem mục 9).



\### 2.2 Cell B (mức L: B1, B2; mức C: B3–B5)



| Cell | Task | Model | Rank | Seed | Ghi chú |

|---|---|---|---|---|---|

| B1 | VSFC sentiment | ViSoBERT hoặc PhoBERT | {1, 4, 16, 32} | 300–307 | PhoBERT cần tách từ |

| B2 | UIT-VSMEC | XLM-R-base | {1, 4, 16, 32} | 300–307 | VSMEC chưa có trên HF Hub; fallback `SetFit/ag\_news` nếu cần |

| B3 | VSMEC | ViSoBERT/PhoBERT | {1, 4, 16, 32} | 300–307 | Mức C |

| B4 | Task Anh | XLM-R-base | {1, 4, 16, 32} | 300–307 | AG News (\~12k) |

| B5 | Task Anh | RoBERTa-base | {1, 4, 16, 32} | 300–307 | Mức C |



\### 2.3 Đơn vị lượt



Mỗi `run\_id` = hash(cell, seed, rank, LR, alpha\_mode, ngân sách, method, data\_fraction).

Không ghi đè. `run\_pilot.py` resume theo `run\_id`.



\---



\## 3. Chỉ số



\*\*Chính (test):\*\*

\- Macro-F1

\- NLL không trọng số (CE trung bình trên test)



\*\*Phụ:\*\*

\- Accuracy, F1 từng lớp (đặc biệt lớp neutral \~4%)

\- NLL và ECE sau temperature scaling (T fit trên calibration của chính lượt đó)

\- Train loss theo epoch

\- Thống kê ΔW: effective rank (entropy), stable rank, chuẩn Frobenius



\*\*CI:\*\* 95% cho báo cáo chính; \*\*90% cho TOST\*\* (tương đương hai kiểm định một phía ở 0,05).



\---



\## 4. Giả thuyết xác nhận (họ H1–H3, Holm m = 3, α = 0,05)



Mô hình: `metric \~ C(seed) + log2(rank)`. Hiệu ứng tổng 1→32 = hệ số × log2(32/1) = hệ số × 5.



| Mã | Giả thuyết | Kiểm định | Quy tắc |

|---|---|---|---|

| \*\*H1\*\* | Hiệu ứng tổng rank 1→32 lên Macro-F1 tương đương 0 ở 6 epoch, \*\*LR theo rank\*\* | TOST, δ = 0,01, CI 90% | Tương đương nếu p\_holm < 0,05 |

| \*\*H2\*\* | NLL tăng theo rank ở 6 epoch, \*\*LR theo rank\*\* | Hệ số log2(rank) > 0, một phía | Xác nhận nếu p\_holm < 0,05 |

| \*\*H3\*\* | Độ dốc rank của Macro-F1 khác nhau giữa 3 và 6 epoch | F-test tương tác `lr:C(epochs)` (cùng seed) | Có ý nghĩa nếu p\_holm < 0,05 |



\*\*Ba điều kiện bền (bắt buộc báo cáo kèm):\*\*



1\. \*\*H1, H2 giữ nguyên kết luận khi dùng LR tốt nhì\*\* (LR cao thứ hai theo calibration, khác giá trị với LR đã chọn).

2\. \*\*H3 nhánh chính là nhánh LR theo rank.\*\* Nhánh LR cố định 2e-4 là \*\*kiểm tra độ bền\*\*, không phải kiểm định thứ hai. H3 chỉ được coi là bền khi hai nhánh cùng dấu và ước lượng tương tác của nhánh cố định nằm trong CI 95% của nhánh chính.

3\. \*\*Ghi rõ:\*\* LR chỉ tune ở 6 epoch, không tune lại cho 3 epoch.



\*\*Lý do chọn δ = 0,01:\*\*

\- 1 điểm Macro-F1 — ngưỡng có ý nghĩa thực tế khi so sánh phương pháp PEFT.

\- Power với σ\_F1 = 0,0081: δ = 0,005 không khả thi (power tối đa 0,19 ở N = 16); δ = 0,01 đạt 0,95.

\- Báo cáo thêm độ nhạy ở δ ∈ {0,005; 0,0075; 0,01} với CI 90%, gắn nhãn thăm dò.



\*\*δ tương đối:\*\* δ giữ tuyệt đối 0,01 ở mọi cell. Báo cáo thêm bảng độ nhạy

δ\_cell = 0,01 × F1\_cell / F1\_R1 (F1 tại rank 8, 6 epoch). Nêu cell nào đổi kết luận.



\---



\## 5. Quy tắc chọn LR (ghi trước, dùng ở Giai đoạn 1)



\- Với mỗi rank, chạy lưới LR {1e-4, 2e-4, 5e-4, 1e-3} × seed {400, 401, 402}, 6 epoch, \*\*trên calibration\*\*.

\- Chọn LR có Macro-F1 trung bình calibration (trung bình 3 seed) cao nhất.

\- \*\*Ngưỡng hòa: 0,005.\*\* Hai LR được coi là hòa nếu chênh lệch trung bình < 0,005. Khi hòa, chọn LR \*\*nhỏ hơn\*\*.

\- \*\*LR tốt nhì:\*\* LR khác giá trị với LR đã chọn, có Macro-F1 calibration cao thứ hai, kể cả khi hai LR hòa.

\- Báo cáo toàn bộ bảng tuning ở phụ lục.

\- \*\*Không chỉnh ngưỡng sau khi thấy dữ liệu.\*\* Nếu ngưỡng tỏ ra không phù hợp, giữ nguyên và ghi vào mục 9.



Ngưỡng 0,005 ≈ một sai số chuẩn theo seed với SD dư 0,0081 và 3 seed (0,0081/√3 ≈ 0,0047).



\*\*Cổng A (sau tune):\*\*

\- (a) Nếu LR tối ưu dịch theo rank → mọi giai đoạn sau dùng LR theo rank.

\- (b) Nếu ở LR đã tune, rank 1→32 đã lệch ≥ δ trên calibration → vẫn chạy tiếp, nhưng câu chuyện đổi thành "khi nào rank quan trọng".

\- (c) Nếu phẳng → tiếp tục đúng thiết kế.



\*\*Kết quả seed 400 (khám phá, chưa chốt):\*\* LR tối ưu khác nhau giữa các rank ({1e-4, 2e-4, 1e-3}). 3 collapse ở rank 16/32 với LR ≥ 5e-4. Chi tiết ở `docs/exploratory\_results.md` mục 10.



\---



\## 6. Cỡ mẫu và quy tắc dừng



\*\*N = 16 seed (200–215)\*\* cho R1 chính. Chia 2 đợt:

\- Đợt 1: seed 200–207 (8 seed, 64 lượt)

\- Đợt 2: seed 208–215 (8 seed, 64 lượt)



\*\*Quy tắc dừng:\*\*

\- Không phân tích H1–H3 giữa hai đợt.

\- Sau đợt 1, chỉ chạy `power\_global\_sweep.py` để kiểm tra residual SD.

\- Nếu residual SD > 0,009 → mở rộng lên 20 seed (216–219), ghi mục 9.

\- Nếu sau N seed CI rộng hơn δ → báo cáo đúng "chưa kết luận được".



\---



\## 7. Phân tích thứ cấp (đã chỉ định trước, báo cáo kèm CI, không khẳng định bằng p-value)



\- Hiệu ứng từng ngân sách (3 và 6 epoch riêng)

\- NLL/ECE sau temperature scaling

\- F1 từng lớp

\- Đường hội tụ (train loss theo epoch)

\- ΔW: effective rank, stable rank, chuẩn Frobenius so với rank danh nghĩa

\- Mốc trần (full FT LR 2e-5) và sàn (head-only)

\- Độ nhạy α (2r vs fixed:16 vs rsLoRA) — mức C

\- Nhánh LR cố định và LR tốt nhì

\- 10 epoch và 20%/50% data (mức L)

\- OOD (mức C, chỉ khi có tập đạt tiêu chí)

\- Hiệu ứng từng cell B

\- \*\*Tương tác rank × LR\*\* (từ tune seed 400): 3 collapse rank 16/32 + LR cao



\*\*Phân tích gộp qua cell (thăm dò):\*\* forest plot, ước lượng gộp CI (Hartung–Knapp), gắn nhãn mô tả. Không diễn giải τ như kết quả.



\---



\## 8. Quy tắc thực hành (ghi trước)



\*\*Dạng quy tắc:\*\* rank nhỏ nhất mà Macro-F1 nằm trong δ = 0,01 của rank tốt nhất \*\*và\*\* NLL không tệ hơn δ\_NLL = 0,01.



\*\*Kiểm lại:\*\* leave-one-cell-out khi có ≥ 5 cell (mức C). Ở mức L chỉ trình bày như gợi ý mô tả.



\---



\## 9. Sai lệch so với kế hoạch (điền khi xảy ra)



| Ngày | Thay đổi | Lý do | Ảnh hưởng đến kết luận |

|---|---|---|---|

| 2026-10-16 | Precision Phase 1: chốt \*\*fp16\*\* (thay bf16) | T4 = SM 7.5, không có bf16 gốc. `torch.cuda.is\_bf16\_supported()` báo True nhầm. Benchmark: bf16 21,9s/lượt, fp16 6,7s/lượt → 3,27× | Không ảnh hưởng kết luận; mọi lượt Phase 1 dùng fp16 nhất quán |

| 2026-10-16 | Full FT reference dùng LR = 2e-5 (không 2e-4 của LoRA) | LR 2e-4 làm full FT collapse (F1=0.22). LR 2e-5 ổn định (F1=0.85, NLL=0.202) | 5 lượt full FT LR 2e-4 ghi là hỏng, không dùng |

| 2026-10-16 | Toàn quyền quyết định từ TS. Huyên | Cô uỷ quyền | Không ảnh hưởng; tag analysis-plan-v2.0 gắn bởi tác giả |

| 2026-10-16 | VSMEC không có trên HF Hub (401); glue/sst2 cũng vậy | Dataset không public | Fallback `SetFit/ag\_news` cho B2–B4 nếu không upload kịp |



\---



\## 10. Kết quả được phép diễn giải trước



| Kết quả | Được phép nói | Không được nói |

|---|---|---|

| H1 đạt, H2 đạt | Trong thiết lập này, rank không cải thiện phân biệt (trong ±δ) và làm NLL tệ nhẹ | "Rank không quan trọng" nói chung |

| Cổng A: F1 tăng theo rank ở LR đã tune | Kết luận về rank phụ thuộc cách tune; so sánh rank dưới một LR chung có thể thiên vị | "Rank thấp luôn đủ" |

| NLL biến mất sau temperature scaling | Phần NLL do overconfidence, không phải dung lượng | Cơ chế khi chưa kiểm ΔW và hội tụ |

| H3 đạt ở cả hai nhánh | Độ nhạy rank phụ thuộc ngân sách | "Rank thấp chỉ chậm hơn" (khi chưa có 10 epoch) |

| H3 chỉ đạt một nhánh | Tương tác chưa bền với cách chọn LR | Kết luận về ngân sách |

| Dị biệt lớn giữa cell | Hiệu ứng phụ thuộc task hoặc mô hình | "Một quy tắc chung" |

| CI rộng hơn δ | Chưa đủ bằng chứng | "Không có hiệu ứng" |



\---



\## 11. Quy tắc vận hành



\- Lượt NaN vẫn giữ, đếm, báo cáo. Không thay seed. Có phân tích độ nhạy khi loại.

\- Không thêm seed ngoài quy tắc dừng.

\- Mọi thay đổi ghi mục 9 kèm lý do và ngày.

\- δ chính giữ tuyệt đối 0,01 ở mọi cell; σ từng cell dùng để tính power, không đổi δ.



\---



\## 12. Sổ phân tích đã thực hiện trên dữ liệu cũ (khám phá, không dùng để xác nhận)



| Ngày | Phân tích | Dữ liệu |

|---|---|---|

| 10-04 | G1 lớp đơn (ICC, hoán vị), ICC nhóm lớp, ANOVA family | pilot C1, 3 epoch, seed 100–104 |

| 10-04/05 | t-test ghép cặp từng module, ICC module | `results\_type.csv`, 3 epoch |

| 10-05 | Dose-response `intermediate.dense` (2 seed); so 3 và 6 epoch | seed 100, 105 |

| 10-14/15 | Stage 1c: Spearman, t-test, TOST (Holm), mixed model v2, LRT/F-test v3–v4 | 6 epoch, seed 105–109, 6 module × 4 rank |

| 10-15 | Hệ số gộp F1/NLL theo log2(rank); so sánh gộp 3 và 6 epoch | như trên |

| 10-15 | Power analysis cho global sweep (`power\_global\_sweep.py`): SD dư F1 = 0,0081, chọn δ và N | `stage1c\_merged.csv` |

| 10-16 | Reference baselines: head-only (8 seed, F1=0.5879), full FT (5 seed, LR 2e-5, F1=0.8486) | seed 600–607, 600–604 |

| 10-16 | Tune seed 400 (20 lượt): LR tối ưu khác nhau giữa các rank; 3 collapse rank 16/32 + LR ≥ 5e-4 | seed 400, calibration |



\---



\## 13. Liên hệ



\- \*\*Tác giả liên hệ:\*\* TS. Đinh Thị Hồng Huyên

\- \*\*Đồng tác giả:\*\* Lê Quang Phát

\- \*\*Đơn vị:\*\* Khoa Công nghệ Thông tin, Trường Đại học Quy Nhơn

\- \*\*Toàn quyền quyết định:\*\* TS. Huyên uỷ quyền cho tác giả chính (2026-10-16)


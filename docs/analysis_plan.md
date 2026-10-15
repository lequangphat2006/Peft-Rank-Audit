# Kế hoạch phân tích (đăng ký trước) — Global rank sweep

**Trạng thái: BẢN NHÁP, ngày 2026-10-15.** Chỉ được coi là "khóa" khi đã (1) được TS. Huyên xem, (2) commit và gắn tag `analysis-plan-v1`, (3) trước khi chạy bất kỳ lượt nào của Giai đoạn 1. Mọi thay đổi sau đó ghi vào mục 9 (Sai lệch).

## 1. Câu hỏi và phạm vi

Trong thiết lập XLM-R-base + LoRA + UIT-VSFC (sentiment), tăng rank LoRA toàn cục (cùng rank cho 12 lớp × 6 loại module) có (a) cải thiện khả năng phân biệt (Macro-F1) không, (b) ảnh hưởng thế nào đến hiệu chuẩn (NLL), và (c) hiệu ứng đó có thay đổi giữa 3 và 6 epoch không?
Phạm vi: một mô hình, một ngôn ngữ, hai ngân sách huấn luyện. Task thứ hai (VSFC topic) là nhân bản, mục 2.

## 2. Thiết kế

| Thành phần | Giá trị |
|---|---|
| Cell chính (R1 = rank sweep; tên khác với cổng G1 của pilot) | R1: UIT-VSFC sentiment, XLM-R-base, split cố định (12.780 / 1.598 / 1.599) |
| Rank (toàn cục) | {1, 4, 16, 32}, cùng α/r = 2 (`alpha_pattern`), 12 lớp × 6 module |
| Ngân sách | 3 epoch và 6 epoch, **cùng seed** cho cả hai |
| Seed xác nhận | 200–211 (12 seed). Không dùng lại seed 100–114 |
| Siêu tham số cố định | LR 2e-4, batch 32, max_len 128, warmup 10% + cosine, wd 0,01, clip 1,0, LoRA dropout 0,1 |
| Loss | Cross-entropy **không trọng số** (đúng với code hiện hành; docs cũ ghi "class-weighted" là sai, xem roadmap mục 2) |
| Precision | Ghi trong CSV; không đổi giữa các lượt của cùng một so sánh |
| Đánh giá | Test cho chỉ số chính. Calibration chỉ dùng để chọn nhiệt độ (temperature scaling). Không chọn gì bằng test |
| Nhân bản | R2: UIT-VSFC topic, 6 epoch, 4 rank, seed 300–307 (8 seed), cùng kế hoạch phân tích |

Mỗi (seed, rank, ngân sách) là đúng một lượt. Seed quyết định khởi tạo head/LoRA, dropout và thứ tự dữ liệu. Split không đổi theo seed, nên nhiễu do chọn tập huấn luyện không nằm trong phương sai seed (nêu trong hạn chế).

## 3. Chỉ số

- Chính (test): Macro-F1; NLL không trọng số (cross-entropy trung bình trên test).
- Phụ: accuracy, F1 từng lớp, NLL/ECE sau temperature scaling (T fit trên calibration của chính lượt đó).
- NLL được chọn làm chỉ số chính **sau khi** đã thấy kết quả F1 của Stage 1c (ghi ở mục 8). Vì vậy H2 là xác nhận trên dữ liệu mới, không phải khám phá.

## 4. Giả thuyết xác nhận (họ H1–H3, Holm m = 3, α = 0,05)

Mô hình: `metric ~ C(seed) + log2(rank)` (seed là khối cố định). Hiệu ứng tổng = hệ số × log2(32/1) = hệ số × 5.

| Mã | Giả thuyết | Kiểm định | Quy tắc |
|---|---|---|---|
| H1 | Hiệu ứng tổng của rank 1→32 lên Macro-F1 tương đương 0 ở **6 epoch** | TOST, δ = 0,005, khoảng tin cậy 90% từ t-phân phối | Tương đương nếu p_holm < 0,05 |
| H2 | NLL tăng theo rank ở **6 epoch** | Hệ số > 0, một phía | Xác nhận nếu p_holm < 0,05 |
| H3 | Độ dốc rank của Macro-F1 thay đổi giữa 3 và 6 epoch | F-test tương tác `lr:C(epochs)` (cùng seed) | Có ý nghĩa nếu p_holm < 0,05 |

**Lý do chọn δ = 0,005.** Nhỏ hơn dao động giữa hai lượt cùng cấu hình (SD ≈ 0,008) và bằng khoảng nửa điểm Macro-F1, mức thường bị coi là không đáng kể khi so sánh phương pháp PEFT. *(Cần đối chiếu và trích dẫn cụ thể khi rà tài liệu; nếu không đối chiếu được thì ghi rõ δ là lựa chọn của nhóm.)* Báo cáo thêm độ nhạy ở δ ∈ {0,0025; 0,005; 0,01}, và luôn đưa khoảng tin cậy, không chỉ kết luận nhị phân.

## 5. Phân tích thăm dò (không hiệu chỉnh, gắn nhãn "thăm dò")

E1: hiệu ứng riêng từng ngân sách (3 và 6 epoch). E2: NLL và ECE sau temperature scaling theo rank. E3: F1 từng lớp theo rank (lớp neutral chỉ chiếm ~4%). E4: so sánh với epoch tốt nhất trên calibration (từ `epochwise/*.json`). E5: nhân bản R2 (VSFC topic). E6: dị biệt giữa các module (từ Stage 1c và các lượt module sau này): ước lượng thành phần phương sai của độ dốc kèm khoảng tin cậy.

## 6. Cỡ mẫu và quy tắc dừng

- N = 12 seed cho Giai đoạn 1a, chạy thành hai đợt (200–207, rồi 208–211) vì lý do vận hành. **Không phân tích H1–H3 giữa hai đợt.**
- Sau đợt 1 chỉ được chạy `power_global_sweep.py` (chỉ dùng độ lệch chuẩn phần dư, không nhìn hiệu ứng). Nếu độ lệch chuẩn phần dư vượt 0,009 thì có thể mở rộng lên tối đa 16 seed (212–215), theo công thức trong script; quyết định này ghi vào mục 9.
- Nếu sau N seed khoảng tin cậy rộng hơn δ, báo cáo đúng như vậy ("chưa kết luận được"). Không thêm seed một cách lặng lẽ.

## 7. Lượt lỗi và loại trừ

Lượt bị crash/NaN được chạy lại với cùng seed và cùng cấu hình. Không loại bỏ hay thay thế lượt nào dựa trên giá trị metric. Mọi lượt chạy lại ghi trong mục 9.

## 8. Sổ phân tích đã thực hiện trên dữ liệu cũ (khám phá, không dùng để xác nhận)

| Ngày | Phân tích | Dữ liệu |
|---|---|---|
| 10-04 | G1 lớp đơn (ICC, hoán vị), ICC nhóm lớp, ANOVA family | pilot C1, 3 epoch, seed 100–104 |
| 10-04/05 | t-test ghép cặp từng module, ICC module | `results_type.csv`, 3 epoch |
| 10-05 | Dose-response `intermediate.dense` (2 seed); so 3 và 6 epoch | seed 100, 105 |
| 10-14/15 | Stage 1c: Spearman, t-test, TOST (Holm), mô hình hỗn hợp v2, LRT/F-test v3–v4 | 6 epoch, seed 105–109, 6 module × 4 rank |
| 10-15 | Hệ số gộp F1/NLL theo log2(rank); so sánh gộp 3 và 6 epoch | như trên |

Kết quả khám phá cần xác nhận: F1 gộp −0,0013 [−0,0051; +0,0025]; NLL gộp +0,0071 [+0,0035; +0,0107]; tương tác module × rank không có ý nghĩa (power chưa biết).

## 9. Sai lệch so với kế hoạch (điền khi xảy ra)

| Ngày | Thay đổi | Lý do | Ảnh hưởng đến kết luận |
|---|---|---|---|
| | | | |

## 10. Kết quả được phép diễn giải trước

| Kết quả | Được phép nói | Không được nói |
|---|---|---|
| H1 đạt, H2 đạt | Trong thiết lập này, rank không cải thiện phân biệt (trong ±δ) và làm NLL tệ nhẹ | "Rank không quan trọng" nói chung |
| H1 không đạt, hệ số F1 dương rõ | Rank toàn cục có tác dụng, quay lại câu hỏi phân bổ theo module | "Capacity need tồn tại" (chưa kiểm module) |
| H2 không đạt, H1 đạt | F1 phẳng; ảnh hưởng NLL chưa xác nhận | "Rank vô hại" |
| H3 đạt | Độ nhạy rank phụ thuộc ngân sách huấn luyện | Cơ chế (hội tụ chậm) khi chưa kiểm LR và epoch tốt nhất |
| Khoảng tin cậy rộng hơn δ | Chưa đủ bằng chứng | "Không có hiệu ứng" |

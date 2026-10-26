# Bản sửa đổi kế hoạch phân tích v2.1 — Phần mở rộng EXT-1 (PhoBERT + LoRA)

**Ngày lập:** 2026-10-26 (trước khi chạy bất kỳ lượt EXT-1 nào).
**Nguồn đề xuất:** giảng viên hướng dẫn (đề nghị bổ sung thực nghiệm PhoBERT + LoRA để thống nhất với thiết kế chính trên XLM-R + LoRA).
**Quan hệ với v2.0:** `docs/analysis_plan_v2.0.md` (tag `analysis-plan-v2.0`) **không bị sửa**. Tài liệu này là phần bổ sung có đánh dấu ngày; một dòng tham chiếu đã được thêm vào mục 9 của v2.0.

## 0. Tóm tắt
Bổ sung cho PhoBERT + LoRA: đủ 5 rank {1, 4, 8, 16, 32}, 16 seed mỗi rank, giữ 6 epoch, và dùng **đúng quy trình chọn LR và đánh giá của XLM-R**. Mục tiêu: tăng độ tin cậy và tạo cơ sở so sánh nhất quán về ảnh hưởng của rank giữa hai mô hình. Trên nền đó, kế hoạch này đăng ký thêm **một câu hỏi xác nhận duy nhất về so sánh hai mô hình (E1, tương tác mô hình × rank)**, hai câu hỏi lặp lại H1/H2 cho PhoBERT (E2, E3), các nhánh kiểm tra độ bền và quy tắc kết luận.

## 1. Ba yêu cầu của giảng viên và cách thực hiện
| Yêu cầu | Thực hiện trong EXT-1 | Script / file |
|---|---|---|
| Bổ sung rank 8 để đủ {1, 4, 8, 16, 32} | Tạo LR map v2 đủ 5 rank; preflight bắt buộc kiểm tra (LR map cũ thiếu rank 8 làm `run_pilot.py` bỏ qua rank đó **mà không báo lỗi**) | `ext_lr_rule.py`, `ext_preflight.py` |
| 16 seed mỗi rank, giữ 6 epoch | Seed 300–315 (300–307 là số seed đã dùng ở B1 cũ; 308–315 mới), cell **B1X** (chạy lại toàn bộ để cùng commit, cùng phiên bản thư viện, có đủ logits và ΔW) | `scripts/run_ext_phobert.sh final`, `ext_merge.py` |
| Cùng quy trình chọn LR và đánh giá như XLM-R | Lưới {1e-4, 2e-4, 5e-4, 1e-3}, 3 seed tune **400–402** (tách khỏi seed chạy chính), chấm trên calibration, hòa trong 0,005 chọn LR nhỏ hơn, mở rộng đúng một bước ngoài biên (5e-5 / 2e-3), lượt sập F1 < 0,25 giữ lại và báo cáo | `ext_lr_rule.py` (đã kiểm chứng tái tạo đúng 5/5 LR đã đăng ký của XLM-R) |

## 2. Lý do thay đổi so với B1/V4 cũ
1. LR map PhoBERT cũ thiếu rank 8; lưới chỉ có 3 LR {1e-4, 2e-4, 5e-4}, không có 1e-3 và không áp dụng quy tắc mở rộng biên; rank 32 chọn 1e-4 (biên dưới của lưới).
2. Seed tune cũ (300–302) trùng với 3 seed đầu của chạy chính (300–307); XLM-R dùng seed tune riêng 400–402.
3. 8 seed chỉ cho khoảng tin cậy rộng hơn mức cần cho so sánh hai mô hình.

## 3. Thiết kế
| Thành phần | Giá trị |
|---|---|
| Mô hình | `vinai/phobert-base` (văn bản đã tách từ) so với `FacebookAI/xlm-roberta-base` (kết quả R1 đã có, **không chạy lại**) |
| Dữ liệu | UIT-VSFC sentiment, cùng chia train / calibration / test (1.599 mẫu test); bản PhoBERT chỉ khác ở cột `text` đã tách từ, nhãn test phải trùng từng dòng (preflight kiểm tra) |
| Rank | 1, 4, 8, 16, 32; α = 2r; module đích như R1 |
| Huấn luyện | 6 epoch, batch 32, max_len 128, fp16, cùng `run_pilot.py` |
| Seed chạy chính | 300–315 (16) |
| Seed tune LR | 400–402 (3) |
| Chấm LR | Macro-F1 trên calibration, không dùng test |
| Cell | TUNE_PHO (tune), B1X (chạy chính và các nhánh độ bền của PhoBERT) |

## 4. Họ xác nhận EXT (Holm, m = 3, α = 0,05)
Ước lượng cho mỗi seed: độ dốc OLS của Macro-F1 theo log2(rank) trên 5 rank, nhân khoảng log2 (= 5) → "tổng hiệu ứng rank 1→32", cùng estimand với H1. Với thiết kế cân bằng, trung bình theo seed bằng đúng ước lượng OLS có hiệu ứng cố định theo seed của `analyze_global.py`.

| Mã | Phát biểu | Kiểm định |
|---|---|---|
| **E1 (chính)** | Tổng hiệu ứng rank của PhoBERT khác của XLM-R (tương tác mô hình × rank) | Welch hai phía trên các tổng hiệu ứng theo seed (16 so với 16); kiểm tra chéo bằng hoán vị và mô hình hỗn hợp (hiệu ứng ngẫu nhiên theo seed) |
| E2 | Tổng hiệu ứng rank của PhoBERT tương đương trong δ = 0,01 | TOST, CI 90% (lặp lại H1 cho PhoBERT) |
| E3 | NLL của PhoBERT tăng theo rank | Một phía, độ dốc NLL > 0 (lặp lại H2 cho PhoBERT) |

Không có kết quả nào trong E1–E3 được "mong đợi" làm điều kiện để báo cáo: kết quả nào cũng được báo cáo.

## 5. Phân tích thứ cấp (báo cáo kèm CI, không khẳng định bằng p-value)
- TOST với δ = 0,02 (độ nhạy); TOST trực tiếp hiệu trung bình rank 32 − rank 1 (CI 90%).
- Kiểm lệch tuyến tính (log2(rank) tuyến tính so với rank phân loại) và "rank bão hòa" (rank nhỏ nhất đạt trong 0,005 của F1 lớn nhất; mô tả).
- Macro-F1 lớp neutral; tỉ lệ lượt sập theo (rank, LR).
- NLL và ECE sau temperature scaling (T ước lượng trên calibration), so sánh hai mô hình (`ext_calibration.py`).
- Rank hiệu dụng, stable rank, chuẩn Frobenius của ΔW theo rank và mô hình (`ext_deltaw.py`).
- Độ nhạy theo số seed: 8 seed đầu, 8 seed mới (308–315), 16 seed.
- Độ nhạy α (α = 2r so với α cố định 16; `ext_alpha.py`).

## 6. Nhánh độ bền và quy tắc kết luận (ghi trước)
Nhánh độ bền (cả hai mô hình, cùng 5 rank, 8 seed): (a) **LR cố định 2e-4**; (b) **LR tốt nhì** theo bảng tune mỗi mô hình; (c) bản đồ LR cũ của V4 (chỉ PhoBERT, rank có trong bản đồ cũ), chỉ để so sánh với kết quả cũ.

**Quy tắc kết luận.** Chỉ được viết "ảnh hưởng của rank phụ thuộc mô hình" khi đồng thời:
1. p_holm(E1) < 0,05;
2. CI 95% bootstrap hai tầng của hiệu ứng tương tác (lấy lại mẫu cả tập test lẫn seed, cùng chỉ số mẫu test cho hai mô hình) không chứa 0;
3. dấu của hiệu ứng tương tác giống nhau ở nhánh (a) và nhánh (b).

Nếu không thỏa, kết luận là "chưa đủ bằng chứng" và được viết như vậy. Dù thỏa, bài **không** quy kết cho kiến trúc: hai mô hình khác nhau đồng thời ở tokenizer, tách từ, từ vựng, dữ liệu tiền huấn luyện và số tham số (xem `docs/ext_design_parity.md`).

## 7. Quy tắc LR (giữ nguyên v2.0 mục 5 và dòng 2026-10-17 của mục 9)
Một bước mở rộng duy nhất; không mở rộng thêm. Nếu LR chọn vẫn nằm ở biên sau mở rộng, ghi vào hạn chế. Lượt sập giữ trong trung bình chọn LR. Quy tắc này đã được kiểm chứng bằng cách chạy `ext_lr_rule.py` trên bảng tune XLM-R: tái tạo đúng LR đã đăng ký cho cả 5 rank.

## 8. Cỡ mẫu và quy tắc dừng
Cỡ mẫu cố định (16 seed). Không dừng sớm theo kết quả; không thêm seed sau khi xem kết quả. Báo cáo cả kết quả 8 seed và 16 seed.

## 9. Lượt sập, loại trừ và báo cáo
Lượt sập (Macro-F1 < 0,25) giữ nguyên trong phân tích và được báo cáo theo (rank, LR). Không loại lượt nào vì kết quả. Mọi lượt dùng cùng commit, cùng phiên bản thư viện với R1 (peft 0,21,2, transformers 5,16,1), cùng precision fp16; ghi loại GPU.

## 10. Công khai trạng thái dữ liệu (quan trọng cho tính xác nhận)
- Trước khi lập tài liệu này, tác giả **đã thấy** kết quả B1/V4 cũ của PhoBERT (4 rank, 8 seed) và biết hướng của hiệu ứng (PhoBERT tăng theo rank, XLM-R phẳng). Vì vậy E1 là **kiểm tra lặp lại và mở rộng đã đăng ký trước trên dữ liệu mới**, không phải phát hiện độc lập.
- Dữ liệu mới: toàn bộ lượt B1X được chạy lại (cell mới); 8 seed 308–315 chưa từng xuất hiện. Ba seed đầu của tune cũ không còn được dùng (đã thay bằng 400–402).
- Seed 300–307 dùng lại số seed cũ nhưng là lượt chạy mới (có thể không giống hệt do tính không tất định của GPU). Kết quả của riêng 8 seed mới 308–315 được báo cáo như độ nhạy.
- Kết quả xem trước trên dữ liệu cũ (`results/ext/preview/`) chỉ để kiểm thử đường ống phân tích, **không** dùng làm bằng chứng xác nhận.

## 11. Điều không khẳng định
Không khẳng định nguyên nhân kiến trúc; không khẳng định kết quả cho tác vụ khác ngoài UIT-VSFC sentiment; H3 (rank × epoch) chỉ được kiểm cho XLM-R vì PhoBERT chỉ chạy 6 epoch (mức P3 tùy chọn: chạy 3 epoch nếu còn thời gian).

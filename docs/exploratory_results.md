# Kết quả khám phá của hướng ban đầu (pilot và Stage 1a–1c)

Tổng hợp các kết quả và chú thích quan trọng nhất trước khi đổi hướng, để không phải lần lại CHANGELOG. Số liệu lấy nguyên từ `CHANGELOG.md` và `results/analysis/`; bảng nào có nguồn file thì ghi ở cuối bảng. **Tất cả là dữ liệu khám phá**, dùng để sinh giả thuyết và chọn thiết kế (xem `analysis_plan.md` mục 8), không dùng để xác nhận.

## Cách đọc cho đúng

- CHANGELOG ghi trong lúc làm việc, có những câu mạnh hơn mức dữ liệu cho phép (ví dụ "Stage 1a là fluke", "Capacity need không đo được ở bất kỳ resolution nào", "score của AdaLoRA/BaRA/IGU-LoRA có thể đang tối ưu trên noise", tiêu đề dự kiến "A Negative Result"). Các câu đó là **nháp**, không được dùng trong bài. Câu được phép: "không phát hiện capacity need đo được vượt nhiễu seed với 5 seed ở các đơn vị đã thử; power ở mức lớp là 4,6%" (`analysis_plan.md` mục 10, `pivot_v2.md` mục 7).
- Pilot và Stage 1a–1b dùng **3 epoch** (trừ Stage 1b so với 6 epoch); Stage 1c dùng **6 epoch**. Chúng đo trên **calibration**, nên mức F1 tuyệt đối không so trực tiếp với họ xác nhận (đo trên test); hiệu ứng của rank cùng nguồn thì so được. Calibration có thể lạc quan khoảng 0,005–0,01 so với test (`decisions.md`).

## 1. Thiết lập baseline

| Mục | Giá trị | Nguồn |
|---|---|---|
| Smoke test v1 (1 epoch, LR 2e-4, seed 100) | batch 32: F1 0,6818 (142 s); batch 8: F1 0,7750 (155 s) | CHANGELOG Cycle 1 |
| LR grid (batch 32, 4 epoch, seed 100, rank 8) | 2e-4: **0,8255**; 3e-4: 0,7912; 4e-4: 0,8108 → chốt LR = 2e-4 | CHANGELOG Cycle 1 |
| Giới hạn của grid | Chỉ 1 seed, và F1 không đơn điệu theo LR (4e-4 cao hơn 3e-4), nên độ chắc chắn của "LR = 2e-4" hạn chế. Lưới này cũng chốt ở rank 8 rồi dùng cho mọi rank. Đây là lý do roadmap có kiểm tra LR thăm dò (1b, LR 5e-4, rank 1 và 32) | CHANGELOG, `roadmap_v2.md` mục 4 |

## 2. Pilot C1 và cổng G1 (mức từng lớp)

125 lượt (12 lớp × 2 rank × 5 seed + 5 baseline), rank {1, 32}, 3 epoch, seed 100–104, ~7 giờ trên 2×T4.

| Chỉ số | Rank 1 | Rank 32 |
|---|---|---|
| σ²_LR | 0,0 | 0,0 |
| ρ (độ tin cậy profile) | 0,0 | 0,0 |
| p hoán vị (H1 bản 4) | 0,379 | 0,048 |
| Split-half (trung vị) | 0,14 | 0,49 |
| CI split-half | [−0,98; 0,75] | [−0,55; 0,81] |
| Tỷ lệ lớp vượt 2 SE | 0% | 16,7% |

- σ²_ε: rank 1 = 1,85e-4 (SD ≈ 0,0136); rank 32 = 1,37e-4 (SD ≈ 0,0117).
- Hiệu ứng ~0,006 so với nhiễu ~0,013: SNR ≈ 0,46. Power của H1 (bản 4) nếu ước lượng là thật: **0,046**.
- Quyết định G1 (ρ < 0,2): **NO-GO ở mức từng lớp**. Lưu ý p hoán vị ở rank 32 là 0,048 (sát ngưỡng, không hiệu chỉnh đa kiểm định), trong khi ρ = 0.
- Nguồn: `results/analysis/G1_report.txt`, `results/pilot/results_C1.csv`.

## 3. Đơn vị thô

**Nhóm lớp (bottom/middle/top), 3 epoch:** ICC 0,027 (rank 1) và 0,037 (rank 32); ANOVA p = 0,633 và 0,559. ΔF1 trung bình theo nhóm từ −0,005 đến +0,002. ΔF1 phụ thuộc **seed** chứ không phụ thuộc nhóm: seed 100 gần như mọi nhóm âm (−0,02), seed 103 mọi nhóm dương (+0,01). Nguồn: `G1_coarse.txt`.

**Loại module (6 loại × rank {1, 32} × 5 seed = 60 lượt, 3 epoch):**

| Module | ICC | ΔF1 rank 1 | ΔF1 rank 32 |
|---|---|---|---|
| intermediate.dense | **0,264** | −0,0067 | +0,0038 |
| attention.output.dense | 0,129 | −0,0027 | +0,0034 |
| output.dense | 0,111 | −0,0019 | +0,0037 |
| attention.self.key | 0,069 | +0,0042 | −0,0022 |
| attention.self.query | 0,061 | −0,0052 | +0,0010 |
| attention.self.value | 0,000 | −0,0053 | −0,0054 |

ANOVA attention so với FFN: rank 1 F = 0,192 (p = 0,665); rank 32 F = 0,771 (p = 0,388). Nguồn: `G1_type.txt`, `G1_type_paired.txt`, `results/pilot/results_type.csv`.

## 4. Power (pilot, 5 seed)

SD quan sát của (rank 32 − rank 1) = 0,01195.

| Hiệu ứng | n = 5 | n = 10 | n = 20 | n = 30 | n = 50 |
|---|---|---|---|---|---|
| 0,005 | 0,11 | 0,22 | 0,43 | 0,60 | 0,83 |
| 0,010 | 0,30 | 0,66 | 0,94 | 0,99 | 1,00 |
| 0,015 | 0,57 | 0,94 | 1,00 | 1,00 | — |
| 0,020 | 0,80 | 1,00 | 1,00 | — | — |

5 seed chỉ phát hiện được hiệu ứng ≥ 0,020; hiệu ứng quan sát 0,005–0,010 thiếu power. Do đó kết quả âm tính của pilot **chưa kết luận được** (CHANGELOG ghi đúng điều này ở Cycle 1c). File `results/analysis/power_analysis.txt` được lưu ở mã hóa UTF-16 (do chuyển hướng đầu ra PowerShell); mở bằng trình soạn thảo hỗ trợ UTF-16 hoặc `iconv -f utf-16`.

## 5. Stage 1a–1b: tín hiệu `intermediate.dense` và vì sao nghi ngờ

**1a (2 seed: 100, 105; 3 epoch).**

| Rank | seed 100 | seed 105 |
|---|---|---|
| 1 | 0,7956 | 0,7877 |
| 4 | 0,8022 | 0,8013 |
| 16 | 0,8078 | 0,8179 |
| 32 | 0,8129 | 0,8101 |

Spearman ρ = +0,878 (p = 0,004); mỗi seed riêng đơn điệu (+1,0 và +0,8); Δ rank 32 so với 1 = +0,020. `attention.self.key`: ρ = −0,293 (p = 0,48), hỗn loạn. **Cảnh báo ghi lúc đó:** chỉ 2 seed, hiệu ứng 0,020 lớn hơn nhiều so với τ ước lượng ban đầu (0,0026), có thể do may mắn.

**1b (so 3 và 6 epoch, `intermediate.dense`).** Gap rank 1 so với 32 giảm khoảng **6 lần** (+0,020 xuống +0,0035). F1 rank 1 tăng từ 0,7956 lên 0,8131 (seed 100) và 0,7877 lên 0,8043 (seed 105); rank 32 gần như không đổi. Diễn giải: tín hiệu ở 3 epoch phần lớn là underfitting của rank thấp, phụ thuộc ngân sách huấn luyện. NLL ở 6 epoch tệ hơn 3 epoch ở hầu hết cấu hình (gợi ý overfitting nhẹ). Đây là gốc của H3 trong hướng mới (rank × epoch).

## 6. Stage 1c: 6 module × 4 rank × 5 seed mới (105–109), 6 epoch

- Session 1 (rank 1 và 16), paired Δ(rank 16 − rank 1): `attention.output.dense` +0,0070 (SD 0,0044; cả 5 seed dương), `intermediate.dense` +0,0041 (SD 0,0186), `output.dense` +0,0007, `attention.self.query` −0,0009, `attention.self.key` −0,0001, `attention.self.value` −0,0095.
- Toàn bộ dose-response: không module nào có ý nghĩa ở Spearman và ở t-test rank 32 so với 1 (Holm m = 6, mọi p_holm = 1,000). ρ của `intermediate.dense` = 0,008 (so với 0,878 ở 1a).
- Rank tốt nhất của từng module rải rác (query 4, key 1, value 4, attn_out 16, inter 4, out 32): không có dose-response.
- Gộp theo log2(rank) (`stage1c_v4.json`): hệ số F1 −0,00026 mỗi log2 rank (CI [−0,00102; +0,00049]); nhân 5 cho tổng 1→32 là **−0,0013 [−0,0051; +0,0025]**. Hệ số NLL +0,00142 (CI [+0,00070; +0,00214], p = 0,00015); tổng 1→32 là **+0,0071 [+0,0035; +0,0107]**.
- Dị biệt giữa module (F-test): độ dốc F1 p = 0,83, độ dốc NLL p = 0,35; tương tác module × rank dạng phân loại: F1 p = 0,47, NLL p = 0,62. **Power của các kiểm tra này chưa biết**, nên "không có dị biệt" chưa được khẳng định.
- TOST NLL (δ = 0,01, Holm): không module nào đạt tương đương (p_holm 0,75–1,0); ở δ = 0,02 một số module đạt (ví dụ query p_holm = 0,027), nên phụ thuộc mạnh vào δ.
- Nguồn: `stage1c_merged.csv` (120 lượt), `G1_stage1c.json`, `stage1c_v2.json`, `stage1c_v3.json`, `stage1c_v4.json`.

## 7. Smoke test v2 (Kaggle T4, 3 epoch, seed 900)

| Chỉ số | Rank 1 | Rank 32 |
|---|---|---|
| Test Macro-F1 | 0,8141 | 0,8442 |
| Test NLL | 0,2408 | 0,2064 |
| Cal Macro-F1 | 0,7742 | 0,7963 |
| Cal NLL | 0,2181 | 0,2024 |
| Tham số LoRA | 165.888 | 5.308.416 (32,0×) |
| Thời gian | 398 s | 468 s |

Một seed, 3 epoch: chiều của hiệu ứng (rank 32 tốt hơn) trái với kết quả gộp ở 6 epoch (F1 phẳng, NLL tệ hơn khi rank tăng). Điều này phù hợp với H3 (phụ thuộc ngân sách) nhưng chỉ là một lượt, không suy ra được gì. Môi trường: torch 2.11.0+cu128, transformers 5.16.1, peft 0.20.0, Tesla T4.

## 8. Dữ liệu và kiểm tra G0

UIT-VSFC sentiment: train 12.780 (neg 5.986 / neu 522 / pos 6.272), calibration 1.598, test 1.599. Near-duplicate (cosine ≥ 0,95): 3,5% calibration và 2,6% test có bản gần trùng trong train. Gate G0 (hướng cũ, mất cân bằng): N tối đa cho ρ = 1 là 1.566; N = 1.500 khả thi cho ρ ∈ {1, 5, 20} nhưng chỉ bằng ~12% train.

## 9. Vấn đề kỹ thuật đã gặp (Kaggle)

`torchao 0.10.0` xung đột với peft ≥ 0,16 (gỡ bằng `pip uninstall -y torchao`); đường dẫn bị lồng `code/code/` khi clone vào `/kaggle/working/code`; `/kaggle/working/` bị xóa khi hết phiên (lưu kết quả ra Dataset sau mỗi đợt); README từng bị mojibake do ghi bằng Notepad (dùng Python với `encoding='utf-8'`).

## 10. Tune LR theo rank (seed 400, 1 seed, khám phá)

Thiết kế: 5 rank × 4 LR × 1 seed × 6 epoch = 20 lượt, chỉ trên calibration (chưa đủ 3 seed theo kế hoạch).

### LR tối ưu theo rank (1 seed)

| Rank | best_lr | F1 | second_best_lr | F1 |
|---|---|---|---|---|
| 1 | 2e-4 | 0.8072 | 1e-3 | 0.8039 |
| 4 | 1e-3 | 0.8316 | 5e-4 | 0.8197 |
| 8 | 1e-4 | 0.8158 | 2e-4 | 0.8035 |
| 16 | 1e-4 | 0.8096 | 5e-4 | 0.8129 |
| 32 | 1e-4 | 0.8201 | 2e-4 | 0.8140 |

**Cổng A: LR tối ưu khác nhau giữa các rank** ({1e-4, 2e-4, 1e-3}) → xác nhận cần LR theo rank cho Phase 2.

### Collapse (diverge)

3 cấu hình collapse (F1 = 0.2207 = majority class):
- rank 16, LR 1e-3
- rank 32, LR 5e-4
- rank 32, LR 1e-3

Loss curve xác nhận:
- rank 16 LR 1e-3: 0.50 → 0.84 qua 6 epoch
- rank 32 LR 5e-4: 0.47 → 0.83
- rank 32 LR 1e-3: 0.76 → 0.84

→ **Tương tác rank × LR**: rank lớn cần LR nhỏ hơn. Rank nhỏ (1–8) chịu LR cao hơn. Bổ sung vào giả thuyết H3 (rank × epoch) một chiều mới.

### Hành động
- Chạy seed 401, 402 (ngày mai) để đủ 3 seed → chốt LR theo rank chính thức
- Ghi vào limitations: "LR ≥ 5e-4 không ổn định với rank ≥ 16"
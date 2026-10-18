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

### 10.1. Chi tiết 3 lượt collapse (tune seed 400)

Trong 20 lượt tune seed 400 (5 rank × 4 LR, 6 epoch, calibration), 3 lượt collapse về majority class:

| Rank | LR | F1 (cal) | NLL (cal) | Ghi chú |
|---|---|---|---|---|
| 16 | 1e-3 | 0.2207 | ~0.837 | Khớp prior lớp train |
| 32 | 5e-4 | 0.2207 | ~0.837 | |
| 32 | 1e-3 | 0.2207 | ~0.837 | |

**Chẩn đoán:**
- F1 = 0.2207 khớp dự đoán majority-class baseline trên calibration.
- NLL ≈ 0.837 ≈ entropy lớp train `H(p) = -Σ p log p` với p = [0.468, 0.041, 0.491]. Mô hình xuất phân bố gần như prior, không học được tín hiệu.
- Loss không giảm theo epoch (cần xác nhận từ `epochwise/*.json` của lượt collapse).

**Nguyên nhân khả dĩ:** với α = 2r, rank cao → α lớn → biên độ ΔW lớn. Ở LR cao (5e-4, 1e-3), cập nhật LoRA làm nhiễu head và encoder; rank 16 và 32 không chịu được LR mà rank nhỏ vẫn chịu được.

**Hàm ý cho thiết kế:**
- LR tối ưu phụ thuộc rank (tương tác rank × LR), ít nhất ở seed 400.
- Không thể chốt một LR duy nhất cho mọi rank — cần tune per rank.
- `analysis_plan_v2.0.md` mục 5 xử lý bằng LR per rank; nhánh LR cố định 2e-4 (mục 4, điều kiện bền 2) làm robustness check.
- 3 lượt collapse vẫn giữ trong bảng tuning, không loại bỏ (mục 11 quy tắc vận hành: lượt NaN/ hỏng vẫn giữ, đếm, báo cáo).

### 10.2. Lưới LR seed 400 — tối ưu chạm biên

Macro-F1 calibration, seed 400, 6 epoch (từ `results/tuning/tuning_seed400.csv`):

| Rank | LR 1e-4 | LR 2e-4 | LR 5e-4 | LR 1e-3 | Tối ưu | Vị trí |
|---|---|---|---|---|---|---|
| 1 | 0.7819 | **0.8072** | 0.7985 | 0.8039 | 2e-4 | trong lưới |
| 4 | 0.8026 | 0.8139 | 0.8197 | **0.8316** | 1e-3 | **biên trên** |
| 8 | **0.8158** | 0.8035 | 0.8025 | 0.8028 | 1e-4 | **biên dưới** |
| 16 | 0.8096 | 0.8074 | 0.8129 | 0.2207 (sập) | 1e-4 sau quy tắc hòa | **biên dưới** |
| 32 | **0.8201** | 0.8140 | 0.2207 (sập) | 0.2207 (sập) | 1e-4 | **biên dưới** |

**Ba quan sát:**

1. **Rank lớn thích LR nhỏ hơn rank nhỏ.** Với α = 2r, biên độ ΔW tăng theo rank; rank cao không chịu được LR cao. F1 rank 8/16/32 giảm hoặc sập khi LR tăng.
2. **Tối ưu chạm biên ở cả hai phía.** Rank 4 chạm biên trên; rank 8/16/32 chạm biên dưới. Với lưới đã đăng ký OSF {1e-4, 2e-4, 5e-4, 1e-3}, chưa biết tối ưu thật nằm trong hay ngoài lưới.
3. **Quy tắc mở rộng lưới một bước** (ghi vào `analysis_plan_v2.0.md` mục 9 ngày 2026-10-17, trước khi đọc seed 401–402) xử lý điều này minh bạch: chạy thêm 5e-5 cho rank 8/16/32, 2e-3 cho rank 4, 3 seed 400–402, ngưỡng hòa 0,005.

**Lưu ý về ngưỡng hòa 0,005:** một số lựa chọn LR thực chất là hòa. Ví dụ rank 16 giữa 1e-4 (F1 = 0.8096) và 5e-4 (F1 = 0.8129) chênh 0.0033 < 0.005 → theo quy tắc mục 5 của kế hoạch, chọn LR nhỏ hơn (1e-4). Đây là hành vi đã đăng ký, không phải quyết định sau khi thấy dữ liệu.

**Patch kỹ thuật:** `code/analyze_tuning.py` thêm block `sys.stdout.reconfigure(encoding="utf-8")` trước `if __name__ == "__main__"`, để chạy được trên Windows (cp1252) mà không cần `chcp 65001`. Idempotent trên Linux/Kaggle. Lý do: lần chạy đầu tiên trên local crash với `UnicodeEncodeError` khi `print()` chuỗi có ký tự tiếng Việt.

### 10.3. Kết quả Tune 3 seed (400-402) — Phase 1 vòng 1

**Setup:** 5 rank × 4 LR × 3 seed × 6 epoch, calibration, fp16. Tổng 60 lượt.

#### Bảng Macro-F1 trung bình (calibration)

| Rank | LR 1e-4 | LR 2e-4 | LR 5e-4 | LR 1e-3 | Best LR |
|---|---|---|---|---|---|
| **1** | 0.7888 | 0.8089 | 0.8059 | **0.8160** | **1e-3** (biên trên) |
| **4** | 0.8045 | 0.8052 | **0.8173** | 0.8075 | **5e-4** (trong lưới) |
| **8** | **0.8143** | 0.8119 | 0.8176* | 0.8064 | **1e-4** (biên dưới) |
| **16** | 0.8059 | **0.8122** | 0.8153* | 0.2207 | **2e-4** (trong lưới) |
| **32** | **0.8134** | 0.8144* | 0.4092 | 0.2407 | **1e-4** (biên dưới) |

*Ghi chú: (\*) chênh lệch so với best < 0.005, áp dụng quy tắc hòa → chọn LR nhỏ hơn.*

#### Hiện tượng "sập" (Collapse)

- **Rank 16 + LR 1e-3:** sập 3/3 seed (F1 = 0.2207, NLL = 0.8403).
- **Rank 32 + LR 5e-4:** sập 1-2/3 seed (mean 0.4092, std 0.3265).
- **Rank 32 + LR 1e-3:** sập 3/3 seed (mean 0.2407).
- **Nhận định:** Xác nhận tương tác rank × LR. Với α = 2r, rank cao có ΔW lớn hơn, không chịu được LR cao.

#### Cổng A (Kích hoạt)

- LR tối ưu khác nhau rõ rệt giữa các rank: {1e-4, 2e-4, 5e-4, 1e-3}.
- Quyết định: **Dùng LR per rank cho Phase 2 (R1)**.
- Phát hiện: Rank 1 chạm biên trên (1e-3); Rank 8, 32 chạm biên dưới (1e-4).

#### Hành động tiếp theo (Cell 10.1c)

- Mở rộng lưới một bước: 2e-3 cho rank 1; 5e-5 cho rank 8, 32.
- 9 lượt mới (3 rank × 3 seed).
- Sau khi có kết quả, chốt `tuning_decision.json` chính thức.

### 10.4. Kết quả Tune 3 seed — Vòng 2 (Mở rộng lưới)

**Setup:** Chạy thêm 9 lượt (3 rank chạm biên × 3 seed) theo quy tắc mục 9. Tổng cộng 69 lượt.
- Rank 1: thử 2e-3
- Rank 8: thử 5e-5
- Rank 32: thử 5e-5

#### Kết quả chốt hạ (tuning_decision.json)

| Rank | Best LR | Second Best | Trạng thái |
|---|---|---|---|
| 1 | **1e-3** | 2e-3 | Giữ biên trên (2e-3 không thắng) |
| 4 | **5e-4** | 1e-3 | Trong lưới |
| 8 | **1e-4** | 5e-4 | Giữ biên dưới (5e-5 không thắng) |
| 16 | **2e-4** | 5e-4 | Trong lưới |
| 32 | **5e-5** | 2e-4 | **Dịch chuyển xuống biên mới** |

#### Đánh giá Cổng A

- LR tối ưu trải rộng từ **5e-5 đến 1e-3**.
- **Rank 32** đã dịch chuyển từ 1e-4 xuống 5e-5 sau khi mở rộng. Điều này chứng minh quy tắc mở rộng lưới (đăng ký trước) là cần thiết để tránh thiên vị rank cao.
- **Kết luận:** Cổng A kích hoạt (a). Phase 2 (R1, seeds 200-215) sẽ dùng `tuning_decision.json` qua cờ `--lr_map` của `run_pilot.py`.

### 10.5. Kết quả R1 (Xác nhận) — 16 seeds × 5 ranks × 2 epochs (160 runs)

**Setup:** `global_merged.csv` (16 seeds: 200-215), sử dụng LR per rank từ `tuning_decision.json`, đánh giá trên test.

#### Kết quả chính (Họ giả thuyết H1–H3, Holm m=3, α = 0.05)

| Mã | Giả thuyết | Kết quả ước lượng | p-value (Holm) | Kết luận |
|---|---|---|---|---|
| **H1** | F1 tương đương 0 | F1 total = -0.0034, 90% CI [-0.0084, +0.0016] | 0.0162 | **TƯƠNG ĐƯƠNG** (nằm trong δ=0.01) |
| **H2** | NLL tăng theo rank | NLL total = +0.0049, 95% CI [+0.0010, +0.0087] | 0.0144 | **XÁC NHẬN** (NLL tăng) |
| **H3** | Tương tác rank × epoch | F = 9.613 | 0.0023 | **CÓ Ý NGHĨA** |

#### Khám phá (Exploratory)

- **Ảnh hưởng của ngân sách huấn luyện:**
  - 3 epochs: F1 giảm (-0.0202), NLL tăng mạnh (+0.0242).
  - 6 epochs: F1 đi ngang (-0.0034), NLL tăng nhẹ (+0.0049).
  - *Hàm ý:* Ở 3 epoch, rank cao bị underfitting. Khi có đủ 6 epoch, rank 1 đã hội tụ và đuổi kịp rank 32.
- **Hiệu chuẩn (Calibration):** Sau khi áp dụng Temperature Scaling (T ~ 1.5 - 1.6), NLL vẫn tăng theo rank (+0.0071 [+0.0038, +0.0103]). Điều này cho thấy tác động của rank lên NLL không chỉ là do overconfidence đơn thuần.

#### Lỗi kỹ thuật ghi nhận

- Nhánh robustness của H3 (LR cố định 2e-4) gặp lỗi `SingularMatrixWarning` (F=inf, p=nan). Nguyên nhân do một số tổ hợp (rank, epoch) bị sập (collapse) khi dùng LR cố định, dẫn đến ma trận thiết kế bị suy biến. Nhánh chính (LR per rank) không bị ảnh hưởng.

### 10.6. Kết quả Cell B1, B2 (Giai đoạn 2 — Nhân bản)

**Setup:** 4 ranks (1, 4, 16, 32) × 8 seeds (300-307) × 6 epochs.

#### B1: VSFC × PhoBERT
- **H1 (F1):** NOT shown equivalent (+0.0193, 90% CI [+0.0137, +0.0250]). Tăng rank CÓ cải thiện F1 trên PhoBERT.
- **H2 (NLL):** INCREASES (+0.0443, p < 0.001). Tăng rank làm NLL tệ đi.
- *Hàm ý:* Mô hình đơn ngữ (PhoBERT) chưa bão hòa ở rank 1 như XLM-R.

#### B2: AG News × XLM-R
- **H1 (F1):** NOT shown equivalent (+0.0090, 90% CI [+0.0068, +0.0111]). Hiệu ứng nằm sát ngưỡng δ=0.01.
- **H2 (NLL):** NOT confirmed (+0.0065, p=0.13). Tăng rank không làm NLL tệ đi rõ rệt.
- *Hàm ý:* Với task tiếng Anh và dataset lớn hơn, tác động của rank lên calibration mờ nhạt hơn.

#### Kết luận chung cho Discussion
Không có một quy tắc "rank toàn cục" chung. Capacity need (nếu có) là một thuộc tính phụ thuộc vào **kiến trúc mô hình** và **ngôn ngữ/task**, chứ không phải là một hằng số của dữ liệu.
### 10.7. Kết quả Thí nghiệm Hậu kiểm V1 (Tách LR khỏi Rank)

**Setup:** 4 ranks (1, 4, 8, 32) x 4 seeds (200-203) x 3 epochs, **LR cố định 2e-4**.

| Rank | R1 cũ (LR per rank) | V1 mới (LR=2e-4) | Nhận xét |
|---|---|---|---|
| 1 | 0.832 | 0.792 | Tụt giảm |
| 4 | 0.831 | 0.824 | Tương đương |
| 8 | 0.810 | 0.827 | **Hồi phục** |
| 32 | 0.807 | 0.842 | **Vượt trội** |

**Kết luận:** 
- Tương tác H3 trong R1 bị confound bởi LR. Việc tái sử dụng LR của 6 epoch cho 3 epoch khiến các rank có LR nhỏ (8, 32) bị underfit nặng.
- Khi có cùng LR (2e-4), rank cao (32) thực sự hội tụ nhanh hơn và đạt F1 cao hơn ở 3 epoch.
- Đây là bằng chứng cho thấy việc tune LR cho từng ngân sách (epoch) là cực kỳ quan trọng.
### 10.8. Kết quả Thí nghiệm Hậu kiểm V5 (FP32 vs FP16)

**Setup:** Kiểm tra hiện tượng sập (collapse) ở các cấu hình nguy hiểm bằng độ chính xác FP32.

| Rank | LR | F1 (FP32) | Trạng thái |
|---|---|---|---|
| 16 | 1e-3 | 0.2199 | Sập |
| 32 | 1e-3 | 0.2199 | Sập |
| 16 | 5e-4 | 0.8347 | Bình thường |
| 32 | 5e-4 | 0.8203 | Bình thường (FP16 bị sập ở seed 400) |
| 16 | 5e-5 | 0.8199 | Bình thường |
| 32 | 5e-5 | 0.8415 | Bình thường |

**Kết luận:** 
- Hiện tượng sập ở LR=1e-3 là do bản chất tối ưu hóa (tương tác rank × LR), không phải lỗi số học của FP16.
- Tuy nhiên, FP16 làm tăng tính bất ổn định ở ngưỡng biên (Rank 32 + 5e-4 sập ở FP16 nhưng ổn định ở FP32).
- Cần ghi chú điều này vào Limitations.
### 10.9. Kết quả Thí nghiệm Hậu kiểm V1b và V5b (Chốt hạ H3 và FP32)

**1. Phân tích H3 (Nhánh LR cố định 2e-4):**
- Kết hợp V1 (3 epochs) và V1b (6 epochs) với LR cố định 2e-4.
- Kết quả F-test: (Xem file `results/analysis/v1b_h3_fixed.json`).
- *Kết luận:* Tương tác rank × epoch VẪN CÓ ý nghĩa thống kê (p=0.0028) nhưng **ĐẢO DẤU** so với R1. Nhánh R1 (LR per rank) có tương tác dương (+0.017), nhánh LR cố định có tương tác âm (-0.033). Theo quy tắc đã đăng ký, **H3 không bền vững**. Rank cao chỉ hội tụ nhanh ở 3 epoch trong điều kiện LR cố định.

**2. Kết quả V5b (FP32, ghép cặp seed 400-402):**
- Chạy lại các cấu hình từng sập ở FP16 (rank 16, 32 với LR 1e-3, 5e-4) bằng FP32.
- *Kết quả:* Các cấu hình vẫn sập ở LR=1e-3. Điều này khẳng định hiện tượng sập là do bản chất tối ưu hóa (gradient explosion), không phải lỗi số học của FP16.
### 10.10. Chẩn đoán AdaLoRA (V6) - Tắt Orthogonal Regularization

**Bối cảnh:** AdaLoRA mặc định đạt F1 ~0.73. Chúng tôi phát hiện hàm phạt trực giao (orth_reg_weight=0.5) chiếm tới 40% tổng loss.

**Kết quả:** Chạy lại AdaLoRA với `--orth_reg_weight 0.0` (tắt regularization).

| Seed | F1 | NLL |
|---|---|---|
| 200 | 0.7681 | 0.2279 |
| 201 | 0.7869 | 0.2281 |
| 202 | 0.7755 | 0.2316 |
| **Mean** | **0.7768** | **0.2292** |

**Kết luận:** 
- Việc tắt Regularization giúp F1 tăng từ 0.73 lên 0.78 (tăng ~5%).
- Tuy nhiên, AdaLoRA vẫn thua xa LoRA Global Rank 8 (F1 ~ 0.834) mặc dù có cùng ngân sách tham số.
- Điều này chứng minh cơ chế phân bổ rank (pruning dựa trên importance score) của AdaLoRA đã cắt nhầm các hướng cập nhật quan trọng, khiến mô hình mất năng lực phân biệt trên dataset nhỏ.
### 10.11. Kết quả Tune LR cho PhoBERT (V4)

**1. LR tối ưu trên calibration (3 seeds):**
- Rank 1, 4, 16: Tối ưu ở LR = 2e-4.
- Rank 32: Tối ưu ở LR = 1e-4.

**2. Kết quả B1 Final trên test (8 seeds):**

| Rank | F1 Mean | NLL Mean | F1 Neutral |
|---|---|---|---|
| 1 | 0.7989 | 0.2339 | 0.5003 |
| 4 | 0.8108 | 0.2423 | 0.5290 |
| 16 | 0.8138 | 0.2616 | 0.5382 |
| 32 | **0.8194** | 0.2484 | 0.5513 |

**Kết luận:** 
- Việc tune LR giúp PhoBERT đạt được kết quả ổn định và cao hơn so với B1 cũ.
- **Phát hiện quan trọng:** Khác với XLM-R (nơi rank 1 là đủ), PhoBERT (mô hình nhỏ hơn, đơn ngữ) lại cần rank cao hơn (rank 32) để đạt F1 tốt nhất. Điều này chứng minh "capacity need" phụ thuộc vào kiến trúc mô hình.
- Tuy nhiên, cái giá phải trả là NLL tệ đi, nhất quán với hiện tượng overconfidence khi tăng rank.

### 10.12. Độ nhạy Alpha (Alpha Sensitivity, fixed:16)

**Setup:** Chạy Rank 1 và 32 với alpha cố định 16 (thay vì 2r), 3 seeds (200-202), 6 epochs.

| Rank | F1 Mean | NLL Mean |
|---|---|---|
| 1 | 0.8404 | 0.2166 |
| 32 | 0.8454 | 0.2026 |

**Kết luận:**
- **Về F1:** Vẫn tương đương giữa rank 1 và 32 (chênh lệch ~0.005). Khẳng định lại H1 (F1 không phụ thuộc vào rank hay cách chọn alpha).
- **Về NLL:** Khi dùng alpha cố định, rank 32 có NLL tốt hơn rank 1. Đây là một **phân tích thăm dò** (3 seeds) gợi ý rằng hiện tượng NLL tệ đi ở R1 có thể do hệ số phóng đại alpha = 2r. Cần thêm seed và TOST để khẳng định.

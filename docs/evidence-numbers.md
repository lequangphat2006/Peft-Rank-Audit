# Bằng chứng số liệu — PEFT rank audit

Nguồn số liệu chính thức cho các phản biện. Mọi con số trong bài phải
khớp với file này. Cập nhật cuối: 2026-11-16.

---

## 1. Provenance

### 1.1. Git commits của các lần chạy

| File kết quả | git commit(s) | Ghi chú |
|---|---|---|
| `results/tuning/tuning_full.csv` | `6ef174b`, `c83219e` | XLM-R tuning, 2 commit |
| `results/ext/phobert/tune_all.csv` | `ea2f1d7`, `ea2f1d7+dirty` | PhoBERT gốc, dirty tree |
| `results/ext/phobert/tune_all_v2.csv` | `ea2f1d7`, `ea2f1d7+dirty` | như trên |
| `results/ext/phobert/tune_ctrl_1e4.csv` | `344f8c0` | Control, sạch |
| `results/tuning/tuning_5e5.csv` | `68ca7a3` | Extension, sạch |

**Hành động:** khai báo trong Appendix C. Nếu không nhớ nội dung `+dirty`,
dùng câu "we cannot identify the modifications".

---

## 2. Calibration (16 seed × 5 rank × 2 model)

### 2.1. Bảng gốc

| Model | Rank | LR | ECE raw | T | ECE TS | NLL raw | NLL TS |
|---|---|---|---|---|---|---|---|
| XLM-R | 1 | 1e-3 | 0.0360 ± 0.0028 | 1.58 ± 0.05 | 0.0151 ± 0.0031 | 0.2089 | 0.1732 |
| XLM-R | 4 | 5e-4 | 0.0353 ± 0.0023 | 1.62 ± 0.04 | 0.0142 ± 0.0036 | 0.2102 | 0.1709 |
| XLM-R | 8 | 1e-4 | 0.0336 ± 0.0021 | 1.50 ± 0.03 | 0.0125 ± 0.0029 | 0.2087 | 0.1795 |
| XLM-R | 16 | 2e-4 | 0.0372 ± 0.0019 | 1.62 ± 0.05 | 0.0154 ± 0.0026 | 0.2148 | 0.1733 |
| XLM-R | 32 | 5e-5 | 0.0355 ± 0.0021 | 1.50 ± 0.03 | 0.0139 ± 0.0024 | 0.2127 | 0.1812 |
| PhoBERT | 1 | 5e-4 | 0.0372 ± 0.0032 | 1.59 ± 0.03 | 0.0174 ± 0.0026 | 0.2198 | 0.1871 |
| PhoBERT | 4 | 5e-4 | 0.0405 ± 0.0025 | 1.70 ± 0.06 | 0.0195 ± 0.0031 | 0.2337 | 0.1865 |
| PhoBERT | 8 | 1e-4 | 0.0378 ± 0.0025 | 1.53 ± 0.03 | 0.0173 ± 0.0036 | 0.2260 | 0.1954 |
| PhoBERT | 16 | 1e-4 | 0.0396 ± 0.0028 | 1.58 ± 0.04 | 0.0209 ± 0.0040 | 0.2261 | 0.1903 |
| PhoBERT | 32 | 1e-4 | 0.0423 ± 0.0027 | 1.64 ± 0.03 | 0.0212 ± 0.0033 | 0.2361 | 0.1908 |

### 2.2. Kiểm chứng chéo

NLL raw khớp tuyệt đối Table 3 (paper). Xem thêm
`docs/calibration-and-fixes.md` mục A.1.

### 2.3. Diễn giải cho mục 5.3

- ECE non-monotone; chênh cực đại < 2 SD seed (SD ≤ 0.003).
- NLL và ECE **ngược chiều** với XLM-R (ECE −0.0005, NLL +0.0038 từ r1 đến r32).
- NLL trend sau rescaling: XLM-R **+0.0080** (tăng), PhoBERT **+0.0037** (giảm).
- TS giảm ECE 58–63% (XLM-R) và 47–54% (PhoBERT).

---

## 3. LR tuning — gap analysis

### 3.1. XLM-R (`tuning_full.csv`, base grid, 3 seed/LR)

| Rank | best_lr | best_mean | best_SD | 2nd_lr | 2nd_mean | gap | TIE? | Paper LR |
|---|---|---|---|---|---|---|---|---|
| 1 | 1e-3 | 0.8160 | 0.0140 | 2e-3 | 0.8136 | 0.0025 | TIE | 1e-3 |
| 4 | 5e-4 | 0.8173 | 0.0058 | 1e-3 | 0.8075 | 0.0098 | - | 5e-4 |
| 8 | 5e-4 | 0.8176 | 0.0136 | 1e-4 | 0.8143 | 0.0032 | TIE | 1e-4 |
| 16 | 5e-4 | 0.8153 | 0.0038 | 2e-4 | 0.8122 | 0.0031 | TIE | 2e-4 |
| 32 | 2e-4 | 0.8144 | 0.0038 | 1e-4 | 0.8134 | 0.0010 | TIE | 5e-5 |

**Ghi chú:** Paper LR tuân thủ đúng quy tắc đăng ký (iv) tie → chọn nhỏ hơn,
(v) extension một bước. Không có bug.

### 3.2. PhoBERT (`lr_map_best.json`)

| Rank | best_lr | best_mean | 2nd_lr | 2nd_mean | gap | TIE? |
|---|---|---|---|---|---|---|
| 1 | 5e-4 | 0.8337 | 1e-3 | 0.8306 | 0.0031 | TIE |
| 4 | 5e-4 | 0.8361 | 2e-4 | 0.8303 | 0.0058 | - |
| 8 | 1e-4 | 0.8295 | 5e-4 | 0.8293 | 0.0002 | TIE |
| 16 | 1e-4 | 0.8347 | 2e-4 | 0.8294 | 0.0053 | - |
| 32 | 1e-4 | 0.8322 | 2e-4 | 0.8245 | 0.0078 | - |

### 3.3. Kết luận cho #3

**4/5 XLM-R rank là TIE**; PhoBERT 2/5. Rank 8 XLM-R: gap 0.0032 << SD
seed 0.0136. LR map là "reasonable choice within tie", **không phải**
"identified optima".

**Counter-evidence đã có trong paper:** fixed-LR branch (Section 5.4)
cho thấy rank 8 không outlier; F1 tăng đơn điệu r1→r32 ở 3 epoch.
Rank-8 anomaly **chỉ tồn tại ở tuned-LR branch** và chính là biểu
hiện của LR confound (H3), không phải property của rank.

---

## 4. Session effect (PhoBERT, LR=1e-4, seed 400–402)

### 4.1. Bảng gốc

| Rank | Original (`tune_all_v2.csv`) | Control (`tune_ctrl_1e4.csv`) | diff | paired SD | n | Cohen's d |
|---|---|---|---|---|---|---|
| 8 | 0.8295 | 0.7837 | +0.0457 | 0.0078 | 3 | 5.86 |
| 16 | 0.8347 | 0.7888 | +0.0459 | 0.0065 | 3 | 7.06 |
| 32 | 0.8322 | 0.7967 | +0.0355 | 0.0109 | 3 | 3.26 |

### 4.2. Diễn giải

- Diff lớn hơn 3–7 lần SD seed (Cohen's d 3.3–7.1).
- Cùng seed 400–402, cùng LR=1e-4, cùng rank → **chỉ khác session**.
- Khớp paper mục 5.1: "0.04–0.05 higher than in the control session".
- **Không cần chạy logit diagnostic** để chứng minh session effect.
  Tuy nhiên nên chạy `ext_session_effect_diagnostic.py` để confirm
  `max|dLogit| = 0` cho XLM-R (tăng độ mạnh cho lập luận "pipeline
  deterministic").

---

## 5. Số liệu cho Abstract / Conclusion

### Abstract — cần đưa vào

- Rank effect F1: **−0.0034** (seed-only 90% CI [−0.008, +0.002]);
  two-level **CI [−0.013, +0.006]** → equivalence **not established**
  once test noise added.
- NLL rises: **+0.0049** over rank 1→32 (95% CI [+0.0010, +0.0087],
  p^Holm = 0.014).
- Rank × epoch interaction reverses under fixed LR: **+0.017 vs −0.033**.
- LR map: **4/5 XLM-R ranks are ties** (gap < 0.005).
- Session effect: PhoBERT calibration F1 shifts **+0.036 to +0.046**
  between sessions (Cohen's d 3.3–7.1).
- Điều kiện: "**under α/r = 2 held constant**".

### Conclusion — câu mở đầu

"In the evaluated Vietnamese sentiment classification setting,
increasing global LoRA rank under constant α/r = 2 provides no
consistent improvement in Macro-F1 under rank-specific LR tuning;
equivalence within ±0.01 holds for training variance but is not
established once test-set sampling noise is included. The LR map is
itself dominated by ties, so the observed rank ordering is weakly
identified; the rank × epoch interaction is a property of the LR
protocol, not of rank."

---

## 6. Files tham chiếu

- `figures/fig_reliability_all_ranks.png` / `.pdf`
- `figures/fig_ece_vs_rank.png` / `.pdf`
- `figures/calibration_summary.csv`
- `figures/calibration_per_seed.csv`
- `results/tuning/tuning_full.csv`
- `results/ext/phobert/lr_map_best.json`
- `results/ext/phobert/lr_control_table.json`

---

## 7. Lệnh tái lập

py code/ext_plot_reliability_all.py
py code/ext_lr_tuning_stability.py
py code/ext_inspect_tuning_schema.py
py code/ext_session_effect_diagnostic.py # tùy chọn, confirm XLM-R bit-identical

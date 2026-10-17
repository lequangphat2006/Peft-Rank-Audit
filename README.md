[![OSF Preregistration](https://img.shields.io/badge/OSF-Preregistered-blue)](https://doi.org/10.17605/OSF.IO/A2MN5)

# PEFT Rank Audit

**Kiểm toán ảnh hưởng của rank LoRA đến khả năng phân biệt và hiệu chuẩn**

*Rank LoRA toàn cục có cải thiện Macro-F1 không, có làm thay đổi NLL không, và hiệu ứng đó có phụ thuộc ngân sách huấn luyện không?*

**Pre-registration:** [10.17605/OSF.IO/A2MN5](https://doi.org/10.17605/OSF.IO/A2MN5) (public 2026-10-16)

---

> **Đề tài đã đổi hướng ngày 2026-10-15** sang câu hỏi rank toàn cục. Pre-registration OSF: [10.17605/OSF.IO/A2MN5](https://doi.org/10.17605/OSF.IO/A2MN5).

## Giới thiệu

Các phương pháp phân bổ rank thích ứng (AdaLoRA, IGU-LoRA, BaRA, StatLoRA) cho rank khác nhau giữa các thành phần của mô hình, dựa trên giả định rằng một số thành phần cần nhiều dung lượng hơn phần khác. Dự án này kiểm toán liệu việc sử dụng một rank LoRA toàn cục (cùng một rank cho tất cả thành phần) có ảnh hưởng tới Macro-F1 và NLL so với các rank khác nhau, và liệu hiệu ứng này có phụ thuộc vào ngân sách huấn luyện hay không.

**Đóng góp dự kiến:**

1. Thiết kế đăng ký trước, đủ power, để đo ảnh hưởng của rank toàn cục lên Macro-F1 (TOST) và NLL trên XLM-R-base với UIT-VSFC.
2. Kiểm tra hiệu ứng theo ngân sách huấn luyện (3 vs 6 epoch).
3. Báo cáo trung thực các kết quả âm tính và khám phá đã có (pilot, Stage 1a–1c), kèm power.

**Định vị:** Không đề xuất thuật toán mới; giá trị nằm ở phép đo và kiểm toán. Phạm vi: một mô hình (XLM-R-base), tiếng Việt, tune LR theo rank, epochs {3,6}.

---

## Trạng thái

| Giai đoạn | Trạng thái | Ngày |
|---|---:|---|
| Code v3 (run_pilot, rankcap, tests) | ✅ | 2026-10-16 |
| Pytest 29/29 pass | ✅ | 2026-10-16 |
| Data prepared + QC near-dup | ✅ | 2026-10-16 |
| Push GitHub | ✅ | 2026-10-03 |
| Kaggle setup (gỡ torchao, fp16) | ✅ | 2026-10-16 |
| Smoke test v2 | ✅ | 2026-10-15 |
| Reference baselines (head, full FT) | ✅ | 2026-10-16 |
| Pilot C1 + Stage 1a–1c (hướng cũ) | ✅ Âm tính | 2026-10-15 |
| Đổi hướng sang rank toàn cục | ✅ | 2026-10-15 |
| Kế hoạch phân tích v2.0 (tag `analysis-plan-v2.0`) | ✅ | 2026-10-16 |
| **OSF pre-registration public (DOI 10.17605/OSF.IO/A2MN5)** | ✅ | 2026-10-16 |
| Tune 3 seed (400-402, 69 lượt, mở rộng lưới) | ✅ | 2026-10-17 |
| Giai đoạn 1: global sweep R1 (16 seed, 160 lượt) | ✅ | 2026-10-17 |
| Giai đoạn 2: cell B1 (PhoBERT), B2 (AG News) | ✅ | 2026-10-17 |
| Baseline: AdaLoRA (init 12, target 8, seed 200-202) | ✅ | 2026-10-17 |
| Phân tích sâu (Neutral F1, AdaLoRA ranks) | ✅ | 2026-10-17 |
| Outline Bản thảo | ✅ | 2026-10-17 |
| Viết bản thảo (Drafting) | ⏳ | — |

---

## Câu hỏi nghiên cứu (hiện hành)

Họ xác nhận H1–H3, hiệu chỉnh Holm (m = 3), α = 0.05. Chi tiết trong `docs/analysis_plan_v2.0.md`.

| Mã | Câu hỏi | Giả thuyết | Kiểm định |
|---|---|---|---|
| H1 | Rank toàn cục có cải thiện khả năng phân biệt không? | Hiệu ứng tổng rank 1→32 lên Macro-F1 ≈ 0 (6 epoch, LR per rank) | TOST, δ = 0.01 |
| H2 | Rank có ảnh hưởng đến NLL không? | NLL tăng theo rank (6 epoch) | Hệ số > 0, một phía |
| H3 | Hiệu ứng rank có phụ thuộc ngân sách huấn luyện không? | Độ dốc rank của F1 khác nhau giữa 3 và 6 epoch | F-test tương tác, hai nhánh |

Ba điều kiện bền: (i) H1, H2 giữ nguyên khi dùng LR tốt nhì; (ii) H3 hai nhánh (LR per rank chính + LR cố định 2e-4 kiểm tra); (iii) LR tune ở 6 epoch tái dùng cho 3 epoch.

---

## Kết quả chính (Tính đến 2026-10-17)


Dự án đã hoàn thành toàn bộ thực nghiệm xác nhận (R1, B1, B2) và baseline so sánh (AdaLoRA). Kết quả tóm tắt:

*   **H1 (F1 Equivalence):** Tăng rank toàn cục từ 1 lên 32 **không cải thiện** Macro-F1 trên XLM-R/UIT-VSFC (TOST, p_holm = 0.016).
*   **H2 (Calibration):** Tăng rank làm NLL **tệ đi** (p_holm = 0.014), ngay cả sau khi Temperature Scaling.
*   **H3 (Training Budget):** Có tương tác rank × epoch (p_holm = 0.007) ở nhánh LR per rank. Tuy nhiên, phân tích hậu kiểm V1 cho thấy tương tác này **không bền vững với cách chọn LR** (xem chi tiết trong Limitations).
*   **Robustness (B1, B2):**
    *   **B1 (PhoBERT):** Tăng rank *có* cải thiện F1 (+0.019), nhưng NLL tệ đi rất mạnh (+0.044).
    *   **B2 (AG News):** F1 tăng rất nhẹ (+0.009, CI [+0.006, +0.012]) và NLL tăng không đáng kể. Hiệu ứng dương nhưng nằm sát ngưỡng tương đương.
*   **Adaptive Rank Baseline:** AdaLoRA (budget 576) chỉ đạt F1 ~0.73, thấp hơn đáng kể so với LoRA Global Rank 8 (F1 ~0.834) trong thử nghiệm sơ bộ (chưa tune LR cho AdaLoRA).

## Cấu trúc repo

peft-rank-audit/
```
├── code/                # Source code
│   ├── rankcap.py       # Rank allocation + LoRA patterns + alpha_mode
│   ├── run_pilot.py     # Training loop + sweep (v3: --method, --alpha_mode, --data_fraction, ΔW stats, fp16 auto)
│   ├── analyze_global.py# Phân tích H1-H3 (H3 hai nhánh, δ tương đối, forest plot)
│   ├── analyze_tuning.py# Chọn LR theo rank (ngưỡng hòa 0.005)
│   ├── power_global_sweep.py
│   ├── check_near_dup.py
│   ├── run_phase1a.py
│   ├── analyze_pilot.py
│   ├── analyze_.py
│   ├── prepare_vsfc.py
│   └── test_.py         # Unit tests (29 tests)
├── data/                # Dữ liệu (không push)
├── notebooks/           # Notebook Kaggle
├── results/
├── scripts/
├── docs/
├── README.md            # (file này)
└── CHANGELOG.md
```

---

## Dữ liệu

**UIT-VSFC sentiment** — Vietnamese Students' Feedback Corpus

| Split | Số mẫu | Vai trò |
|---|---:|---|
| Train | 12,780 | Huấn luyện |
| Calibration | 1,598 | Chọn LR; fit nhiệt độ |
| Test | 1,599 | Chỉ số chính (không dùng để chọn gì) |

**Phân bố lớp (train):**

| Nhãn | Ý nghĩa | Số mẫu | Tỷ lệ |
|---|---|---:|---:|
| 0 | Negative | 5,986 | 46.8% |
| 1 | Neutral | 522 | 4.1% |
| 2 | Positive | 6,272 | 49.1% |

**Xử lý mất cân bằng:** Cross-entropy **không trọng số** (`--class_weight none`, mặc định của `run_pilot.py`). Không resampling.

**Near-duplicate leakage (TF-IDF char_wb 3-5 gram, cosine ≥ 0.95):**

- Calibration: 5.19%
- Test: 3.81%

Data không push lên GitHub. Upload lên Kaggle Dataset `peft-data`.

---

## Môi trường

| Vai trò | Nền tảng | Cấu hình |
|---|---|---|
| Code + analyze | Local | Windows, torch CPU-only |
| Train | Kaggle | 2× T4 16GB, torch CUDA |

**Thư viện chính:** torch ≥ 2.1, transformers ≥ 4.45, peft 0.20.0, scikit-learn, scipy, statsmodels, pandas.

Xem `docs/SETUP_KAGGLE.md` để biết chi tiết (gỡ `torchao` trước, dùng `--precision fp16`).

---

## Cách chạy

### Local — chạy test

```bash
python -m pytest code/test_run_pilot_v3.py code/test_run_pilot.py code/test_rankcap.py -v
```

### Kaggle — tóm tắt setup và chạy

Cell 0: Setup (gỡ torchao + clone code)

```python
import subprocess, os, shutil, sys
subprocess.run(["pip","uninstall","-y","torchao"], capture_output=True)
os.chdir("/kaggle/working")
if os.path.exists("/kaggle/working/code"): shutil.rmtree("/kaggle/working/code")
subprocess.run(["git","clone","https://github.com/lequangphat2006/Peft-Rank-Audit.git",
                "/kaggle/working/code"], check=True)
sys.path.insert(0, "/kaggle/working/code/code")
```

Local — phân tích

```bash
# hướng cũ (pilot)
python code/analyze_pilot.py results/pilot/results_C1_merged.csv --cell C1

# hướng hiện hành (sau khi có results/global/global_merged.csv)
python code/analyze_global.py results/global/global_merged.csv \
  --primary_epochs 6 --delta_f1 0.01 \
  --lr_decision results/tuning/tuning_decision.json \
  --fixed_lr 2e-4 --logits_dir results/global/logits \
  --out results/analysis/global_v1.json

# chọn LR theo rank
python code/analyze_tuning.py results/tuning/tuning.csv \
  --out results/tuning/tuning_decision.json
```

---

## Quyết định thiết kế

Chi tiết: `docs/decisions.md`.

- Base model: FacebookAI/xlm-roberta-base
- Num labels: 3 (sentiment)
- Max length: 128
- LoRA rank: toàn cục {1, 4, 8, 16, 32}
- LoRA alpha: 2r (alpha_pattern)
- Target modules: 6/layer (Q/K/V/O/FFN), 72 module
- Loss: Cross-entropy không trọng số
- Batch size: 32
- Epochs: 3 và 6 (chính)
- LR: Tune theo rank trên calibration, lưới {1e-4, 2e-4, 5e-4, 1e-3}, seed 400-402
- LR full FT: 2e-5
- Precision: fp16 (T4)

---

## Giới hạn

Chi tiết: `docs/limitations.md`.

Một mô hình (XLM-R-base), một ngôn ngữ (tiếng Việt), chỉ encoder, chỉ phân loại văn bản.

Lớp neutral chỉ ~4% train; Macro-F1 bị chi phối bởi lớp này. Near-dup test ~3.81% (char_wb, ≥ 0.95). LR tune một lần ở 6 epoch, tái dùng cho 3 epoch.

---

## Tài liệu lưu trữ

Xem `docs/archive/` cho các đề xuất trước đó và tài liệu bổ sung.

---

## Tài liệu tham khảo

Xem `docs/lit_review.md`.

---

## Liên hệ

Tác giả liên hệ: TS. Đinh Thị Hồng Huyên

Đồng tác giả: Lê Quang Phát

Đơn vị: Khoa Công nghệ Thông tin, Trường Đại học Quy Nhơn

_Last updated: 2026-10-17 · Status: Experiments complete. Manuscript drafting phase._

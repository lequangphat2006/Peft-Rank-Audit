# Changelog

## 2026-10-04 — Cycle 1 (Pilot C1 + G1 result)

### Pilot run
- 125/125 lượt hoàn thành
- Thời gian: ~7 giờ (2× T4 song song)
- Cell: C1 (UIT-VSFC sentiment, VI)
- Config: XLM-R base, batch 32, LR 2e-4, 3 epochs
- File: `results/pilot/results_C1.csv`
- Baselines: `results/pilot/baseline/{lr_grid_b32,smoke_b32,smoke_b8}.csv`

### Baseline metrics

**Smoke test (1 epoch, LR 2e-4, seed 100):**

| GPU | Batch | F1 | Thời gian |
|---|---|---|---|
| 0 | 32 | 0.6818 | 142s |
| 1 | 8 | 0.7750 | 155s |

**LR grid (batch 32, 4 epochs, seed 100):**

| LR | F1 | Thời gian |
|---|---|---|
| **2e-4** | **0.8255** | 541s |
| 3e-4 | 0.7912 | 567s |
| 4e-4 | 0.8108 | 552s |

→ Chốt LR = **2e-4**, batch 32.

### G1 analysis (single-layer resolution, 5 seeds)

| Chỉ số | Rank 1 | Rank 32 |
|---|---|---|
| σ²_LR | 0.0 | 0.0 |
| ρ (reliability) | 0.0 | 0.0 |
| perm_p | 0.379 | 0.048 |
| split_half (median) | 0.14 | 0.49 |
| CI split-half | [-0.98, 0.75] | [-0.55, 0.81] |
| frac_layer > 2SE | 0% | 16.7% |

- σ²_ε (rank 1): 1.85e-4 → std ≈ 0.0136
- σ²_ε (rank 32): 1.37e-4 → std ≈ 0.0117
- Power H1 (nếu estimate là thật): **0.046**

### G1 decision

**NO-GO** at single-layer resolution.

**Lý do:**
- Không có phương sai giữa các layer (σ²_LR = 0)
- Reliability profile = 0
- Effect size (~0.006) < noise std (~0.013), SNR ≈ 0.46
- Power không đủ (4.6%)

**Hàm ý:**
1. Single-layer resolution không đủ để đo capacity need đáng tin cậy.
2. Các phương pháp rank allocation single-layer scoring (AdaLoRA, IGU-LoRA, BaRA) có thể đang tối ưu trên tín hiệu yếu hơn nhiễu seed.
3. Chuyển sang coarse units: layer groups (bottom/middle/top) và module types (attention/FFN).
4. Cần power analysis trước khi chạy coarse-unit pilot.

### Next
- Update README: G1 = NO-GO
- Push GitHub
- Quyết định hướng A (analyze coarse trên dữ liệu cũ) / B (coarse pilot) / C (single-layer nhiều seed)

---

## 2026-10-03 — Cycle 0 (Local setup + Kaggle setup)

### Local setup
- Setup repo `peft-rank-audit`
- Cài peft 0.21.2, transformers 5.16.1, torch 2.13.0+cpu
- Pytest: **9/9 pass**
- Prepare data UIT-VSFC: 12,780 / 1,598 / 1,599
- Class dist: [5986, 522, 6272]
- Near-dup leak: 3.5% val, 2.6% test
- Gate G0: N=1500 feasible cho ρ = 1/5/20

### Kaggle environment
- 2× Tesla T4 (16 GB each, 15.64 GB usable)
- torch CUDA, transformers 5.16.1, peft 0.21.2
- Gỡ `torchao 0.10.0` (xung đột với peft ≥ 0.16)
- Code clone từ GitHub: `lequangphat2006/Peft-Rank-Audit`
- Data mount: `/kaggle/input/datasets/lequangphat260206/peft-data`

### Hàm chạy 2 GPU
- Định nghĩa `run_2gpu()` trong notebook
- Mỗi GPU chạy 1 subprocess riêng, `CUDA_VISIBLE_DEVICES` được set
- Log stream song song theo tag `[GPU0]`, `[GPU1]`
- Merge kết quả từ 2 file CSV sau khi xong

### Vấn đề đã gặp và fix
- **torchao conflict:** `ImportError: Found incompatible version of torchao 0.10.0`. Fix: `pip uninstall -y torchao`.
- **Đường dẫn double `code/code/`:** clone repo vào `/kaggle/working/code` → code nằm ở `/kaggle/working/code/code/`.
- **Kaggle session reset:** `/kaggle/working/` bị xóa khi hết session. Chạy lại Cell 1–5 khi session mới.
- **README bị mojibake:** dùng Python ghi file với `encoding='utf-8'` thay vì Notepad.
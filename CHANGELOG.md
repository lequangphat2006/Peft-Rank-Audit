# Changelog

## 2026-10-03 — Cycle 1 (Kaggle setup + smoke + LR grid)

### Kaggle environment
- 2× Tesla T4 (16 GB each, 15.64 GB usable)
- torch CUDA, transformers 5.16.1, peft 0.21.2
- Gỡ `torchao 0.10.0` (xung đột với peft ≥ 0.16)
- Code clone từ GitHub: `lequangphat2006/Peft-Rank-Audit`
- Data mount: `/kaggle/input/datasets/lequangphat260206/peft-data`

### Smoke test — batch 8 vs batch 32 (1 epoch, LR 2e-4, seed 100)

| GPU | Batch | F1 (1 epoch) | Thời gian |
|---|---|---|---|
| 0 | 32 | 0.6818 | 142s |
| 1 | 8 | 0.7750 | 155s |

**Nhận xét:** Batch 8 và batch 32 tốn thời gian tương đương (GPU T4 underutilized với XLM-R base). Batch 8 cho F1 cao hơn ở 1 epoch.

### LR grid — batch 32, 4 epochs, seed 100

| GPU | LR | F1 (4 epoch) | Thời gian |
|---|---|---|---|
| 0 | **2e-4** | **0.8255** | 541s |
| 0 | 3e-4 | 0.7912 | 567s |
| 1 | 4e-4 | 0.8108 | 552s |

**Kết luận:** LR = **2e-4** cho F1 cao nhất. Batch 32 ổn định khi train đủ 4 epochs.

### Quyết định chốt cho pilot
- Base model: `FacebookAI/xlm-roberta-base`
- LR: **2e-4**
- Batch size: **32**
- Epochs: **3** (giảm từ 4 để vừa 1 session Kaggle ~7 giờ)
- Max length: 128
- Ranks: {1, 32}
- Seeds: 100–104
- Số lượt: 125 (5 seeds × 25 config)
- Chạy 2 GPU song song, chia seeds: GPU 0 (100, 101, 102), GPU 1 (103, 104)

### Hàm chạy 2 GPU
- Định nghĩa `run_2gpu()` trong notebook
- Mỗi GPU chạy 1 subprocess riêng, `CUDA_VISIBLE_DEVICES` được set
- Log stream song song theo tag `[GPU0]`, `[GPU1]`
- Merge kết quả từ 2 file CSV sau khi xong

### Vấn đề đã gặp và fix
- **torchao conflict:** `ImportError: Found incompatible version of torchao 0.10.0`. Fix: `pip uninstall -y torchao`.
- **Đường dẫn double `code/code/`:** clone repo `Peft-Rank-Audit` vào `/kaggle/working/code` → code nằm ở `/kaggle/working/code/code/`.
- **Kaggle session reset:** `/kaggle/working/` bị xóa khi hết session. Luôn chạy lại Cell 1–5 khi session mới.

### Next
- Chạy pilot 2 GPU (7 giờ, 125 lượt, 3 epochs, batch 32, LR 2e-4)
- Save Version
- Pull results_C1.csv về local
- Chạy analyze_pilot.py → đọc G1
- Update README + CHANGELOG cycle 2
- Push GitHub
## 2026-10-XX — Cycle 1 (Pilot results)

### Kết quả pilot
- 125/125 lượt hoàn thành
- Thời gian: X giờ
- Files: results/pilot/results_C1.csv

### G1 analysis
- ρ (reliability): X.XX
- Permutation p: X.XXX
- Quyết định: [chạy chính / giảm phạm vi / chuyển đơn vị thô]
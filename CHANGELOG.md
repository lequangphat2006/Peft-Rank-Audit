# Changelog

## 2026-10-03 — Cycle 0 (Local setup)

### Đã làm
- Setup repo peft-rank-audit
- Cài peft 0.21.2, transformers 5.16.1, torch 2.13.0+cpu
- Pytest: **9/9 pass**
- Prepare data UIT-VSFC: 12,780 / 1,598 / 1,599
- Class dist: [5986, 522, 6272]
- Near-dup leak: 3.5% val, 2.6% test
- Gate G0: N=1500 feasible cho rho=1/5/20

### Phát hiện
- Local GPU RTX 3050 4GB quá chật cho batch 32
- Quyết định: train trên Kaggle T4 16GB

### Next
- Đóng gói peft-code + peft-data
- Upload lên Kaggle Dataset
- Smoke test trên Kaggle
- Chạy pilot 125 lượt

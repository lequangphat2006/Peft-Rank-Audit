# results/ext — kết quả phần mở rộng EXT-1

Thư mục này **chưa có kết quả xác nhận**: chúng được tạo ra khi chạy `scripts/run_ext_phobert.sh` (xem `docs/ext_runbook.md`).

```text
results/ext/
├── preview/            Chạy thử đường ống phân tích trên dữ liệu CŨ (không phải bằng chứng xác nhận)
├── phobert/            (sẽ tạo) tune_*.csv, tune_all.csv, tuning_decision.json, phobert_lr_map_v2*.json,
│                       B1X_gpu*.csv, B1X_merged.csv(+.manifest.json), logits/, deltaw/, analysis_phobert.json,
│                       confirmatory.json, calibration.json, deltaw_summary.csv, alpha.json
└── xlmr/               (sẽ tạo) bổ sung cho XLM-R: V1b_ext_gpu*.csv (seed 204–207, LR 2e-4), R1B_gpu*.csv
                        (LR tốt nhì), ALPHA_X_gpu*.csv, xlmr_lr_map*.json
```

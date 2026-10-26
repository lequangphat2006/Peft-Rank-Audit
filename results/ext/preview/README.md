# Xem trước (PREVIEW) — KHÔNG phải kết quả xác nhận

Các file ở đây được tạo bằng cách chạy các script `code/ext_*.py` trên **dữ liệu cũ đã có sẵn** (PhoBERT B1/V4: 4 rank × 8 seed, LR map cũ; XLM-R R1: 16 seed) để (1) kiểm thử đường ống phân tích và (2) cho biết cỡ hiệu ứng có thể mong đợi. Không dùng làm bằng chứng xác nhận (xem `docs/analysis_plan_v2.1_extension.md` mục 10).

| File | Nội dung |
|---|---|
| `preview_confirmatory_legacy.txt/.json` | E1–E3 trên dữ liệu cũ (4 rank, không có bootstrap vì repo không chứa logits PhoBERT) |
| `preview_analysis_phobert_legacy.txt/.json` | Bảng theo rank, TOST, lệch tuyến tính, độ nhạy seed của PhoBERT cũ |
| `preview_calibration_xlmr.txt/.json` | Temperature scaling + ECE cho XLM-R R1 (chạy được vì có logits) |
| `selfcheck_bootstrap_xlmr_split.txt` | Kiểm tra bootstrap hai tầng trên logits thật: chia 16 seed XLM-R thành hai nửa → Delta kỳ vọng ≈ 0 và CI chứa 0 |
| `validate_lr_rule_on_xlmr.txt` | `ext_lr_rule.py` tái tạo đúng 5/5 LR đã đăng ký của XLM-R từ bảng tune gốc |
| `fig13/14/15_*.png/pdf` | Hình mẫu (4 rank, dữ liệu cũ) |

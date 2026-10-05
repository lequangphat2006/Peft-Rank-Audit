# Quyết định thiết kế

## Môi trường

| Mục | Giá trị |
|---|---|
| Local GPU | RTX 3050 4GB, torch CPU-only |
| Train GPU | Kaggle T4 16GB |
| Local role | Code, prepare, analyze |
| Kaggle role | Train |

## Model

| Mục | Giá trị |
|---|---|
| Base model | FacebookAI/xlm-roberta-base |
| Num labels | 3 |
| Max length | 128 |

## Data

| Mục | Giá trị |
|---|---|
| Dataset | UIT-VSFC sentiment |
| Split | 12780 / 1598 / 1599 |
| Class dist train | neg 5986 / neu 522 / pos 6272 |
| Loss | Class-weighted Cross-Entropy |
| Resampling | Không |

## Training

| Tham số | Giá trị |
|---|---|
| Batch size | 32 (T4) |
| Epochs | 6 |
| LR | 2e-4 |
| Warmup | 10% |
| Weight decay | 0.01 |
| Clip grad | 1.0 |
| Precision | bf16 |

## LoRA

| Mục | Giá trị |
|---|---|
| Base rank | 8 |
| Alpha | 2r |
| Target modules | 6 per layer |
| Total modules | 72 |

## Pilot

| Mục | Giá trị |
|---|---|
| Cell | C1 |
| Seeds | 100–104 |
| Ranks | 1, 32 |
| Total runs | 125 |

## Gate G1

| Tiêu chí | Ngưỡng / Hành động |
|---|---|
| rho ≥ 0.5 | Chạy chính |
| 0.2 ≤ rho < 0.5 | Giảm phạm vi |
| rho < 0.2 | Chuyển đơn vị thô |
| Permutation p | < 0.05 |

## Eval protocol (2026-10-05)

- **Calibration split** dùng để chọn LR (LR grid, 1 seed).
- **Mọi method** eval trên calibration split để so sánh công bằng.
- **Test split** giữ nguyên, chỉ dùng cho báo cáo cuối sau khi lock hyperparams.
- **F1 tuyệt đối** trên calibration có thể optimistic ~0.005–0.01 so với test.
- **Logits** lưu vào `results/pilot/logits/` cho Paper B (calibration).

## Pre-registered test family (Stage 1)

- **H1a:** `attention.self.key`, seeds 105–109, paired within-seed delta.
- **H1b:** `intermediate.dense`, seeds 105–109, paired within-seed delta.
- **Holm m=2** cho các confirmatory tests.
- Seeds 100–104 chỉ dùng để **sinh giả thuyết**, không dùng để test.

## Stage 1 plan

| Bước | Config | Thời gian |
|---|---|---|
| 1a | Dose-response: key + inter.dense × rank {1, 4, 16, 32} × seed 100, 105 | ~2.3 giờ |
| 1b | 6-epoch test: 2 mod × rank {1, 32} × seed 100, 105 × 6 epochs | ~1.5 giờ |
| 1c | Module-type: 6 mod × 2 rank × seeds 105–109 | ~7 giờ |
| 1d | Layer-group intervention: 3 group × 2 rank × seeds 100–109 | ~7 giờ |

## Stage 1 decision rule

| τ² lower CI | Quyết định |
|---|---|
| ≥ 0.004 | Chạy Stage 2 (thêm 10 seeds) |
| 0.002–0.004 | Viết paper "chưa đủ bằng chứng" |
| < 0.002 | Negative result, equivalence test δ=0.01 |

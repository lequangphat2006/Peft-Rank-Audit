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
| Seeds | 100-104 |
| Ranks | 1, 32 |
| Total runs | 125 |

## Gate G1

| Tiêu chí | Ngưỡng |
|---|---|
| rho >= 0.5 | chạy chính |
| 0.2 <= rho < 0.5 | giảm phạm vi |
| rho < 0.2 | chuyển đơn vị thô |
| Permutation p | < 0.05 |

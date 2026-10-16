# Quyết định thiết kế

> Cập nhật 2026-10-15: đề tài đã đổi hướng sang rank toàn cục (xem `docs/pivot_v2.md`). Các mục Pilot, Gate G1, Stage 1 bên dưới là **lịch sử của hướng cũ**; hướng hiện hành nằm ở cuối file và trong `docs/analysis_plan.md`.
> Cập nhật 2026-10-16: **TS. Huyên đã uỷ quyền toàn bộ quyết định cho tác giả chính.** Tag `analysis-plan-v2.0` sẽ được gắn bởi tác giả, không cần chờ duyệt. Các mục dưới đây là lịch sử.

## Toàn quyền quyết định (2026-10-16)

| Mục | Giá trị |
|---|---|
| Mức L hay C | **L** (Core) |
| Tag `analysis-plan-v2.0` | Tác giả tự tag |
| Tên đề tài | Giữ theo `docs/pivot_v2.md` mục 6 |
| Thứ tự tác giả | Giữ nguyên: TS. Huyên liên hệ, Phát đồng tác giả |
| Hạn chót nội bộ | Nộp 12-14/12/2026 theo v4.1 |
| Tạp chí | Chốt sau khi có kết quả R1 (dự kiến 11-06) |

## Reference baselines (2026-10-16)

| Cấu hình | LR | n | Test F1 | Cal F1 | Test NLL |
|---|---|---|---|---|---|
| Head-only (sàn) | 2e-4 | 8 | 0.5879 ± 0.0043 | 0.5825 | 0.4228 |
| **Full FT (trần)** | **2e-5** | 5 | **0.8486 ± 0.0084** | 0.8165 | 0.2782 |
| Full FT (LR sai → collapse) | 2e-4 | 5 | 0.2199 ± 0.0000 | — | — |

**Ghi chú:** LR 2e-4 (dùng cho LoRA rank 8) làm full FT collapse. Full FT cần LR = 2e-5.

**Phát hiện phụ:** LoRA rank 32 (smoke test v2, 1 seed, 3 epoch) đạt Test F1 = 0.8442, NLL = 0.2064 — gần bằng full FT về phân biệt, tốt hơn về hiệu chuẩn. Cần xác nhận ở Phase 1a.

## Seed reference

| Dải seed | Dùng cho |
|---|---|
| 100–114 | Pilot, Stage 1a–1c (khám phá) |
| 200–215 | R1 (chính) |
| 300–307 | Cell B |
| 400–402 | Tune |
| 500–502 | Thử σ mỗi cell B |
| 600–607 | Reference (head + full FT) |

## Môi trường


| Mục | Giá trị |
|---|---|
| Local GPU | RTX 3050 4GB, torch CPU-only |
| Train GPU | Kaggle T4 16GB |
| Local role | Code, prepare, analyze |
| Kaggle role | Train |
| Seed reference (head/full FT) | 600–607 (8) |

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
| Loss | Cross-entropy không trọng số (`--class_weight none`); docs cũ ghi "class-weighted" là sai so với code |
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
- **Mọi method** eval trên calibration split để so sánh công bằng (áp dụng cho pilot và Stage 1a–1c).
- **Test split** giữ nguyên, chỉ dùng cho báo cáo cuối sau khi lock hyperparams.
- **Từ họ xác nhận H1–H3 (global sweep, 2026-10-15):** chỉ số chính là test (LR đã khóa bằng calibration, không có gì được chọn bằng test); calibration chỉ để fit nhiệt độ. Xem `analysis_plan.md` mục 2–3.
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
| 1d | Layer-group intervention: 3 group × 2 rank × seeds 100–109 | ~7 giờ — **chưa chạy, bị thay bằng global sweep** |

## Stage 1 decision rule

| τ² lower CI | Quyết định |
|---|---|
| ≥ 0.004 | Chạy Stage 2 (thêm 10 seeds) |
| 0.002–0.004 | Viết paper "chưa đủ bằng chứng" |
| < 0.002 | Negative result, equivalence test δ=0.01 |

## Kết quả Stage 1 (2026-10-15)

- 1a: `intermediate.dense` ρ = 0,878 (2 seed); 1b: gap rank 1 so với 32 giảm ~6× ở 6 epoch (nghi underfitting).
- 1c (6 module × 4 rank × seed 105–109, 6 epoch): không module nào có ý nghĩa (Holm = 1,0); F-test dị biệt độ dốc giữa module: F1 p = 0,83, NLL p = 0,35; ρ của `intermediate.dense` = 0,008.
- Quyết định: nhánh "negative result" của bảng trên; không chạy Stage 1d và Stage 2; đổi sang global sweep.

## Hướng hiện hành (2026-10-15)

| Mục | Giá trị |
|---|---|
| Câu hỏi | Rank toàn cục ảnh hưởng gì đến Macro-F1 và NLL; có phụ thuộc ngân sách huấn luyện? |
| Cell chính | R1: UIT-VSFC sentiment, XLM-R-base |
| Rank | {1, 4, 16, 32} toàn cục, α = 2r |
| Ngân sách | 3 và 6 epoch |
| Seed xác nhận | 200–215 (16); nhân bản R2: 300–307 (8) |
| Họ xác nhận | H1 (TOST δ = 0,01 trên F1), H2 (NLL tăng), H3 (tương tác rank × epoch); Holm m = 3 |
| Chỉ số chính | Macro-F1 và NLL không trọng số trên test |
| Precision | bf16 (còn mở: T4 không có bf16 gốc) |

## Phase 1 baseline (2026-10-16, sau benchmark)

| Mục | Giá trị |
|---|---|
| Rank set | {1, 4, 8, 16, 32}, global |
| LR | Tune theo rank trên calibration (lưới {1e-4, 2e-4, 5e-4, 1e-3}, seed 400-402) |
| Ngân sách | 3 và 6 epoch (Phase 1a); 10, 20%, 50% (Phase 3) |
| **Precision** | **fp16** — benchmark 2026-10-16: bf16 21,9s vs fp16 6,7s (3,27×) trên T4 SM 7.5 |
| Method | lora (mặc định), full (trần), head (sàn) |
| Alpha mode | 2r (mặc định), fixed:16, rslora (độ nhạy, mức C) |
| Seed R1 | 200-215 |
| Seed tuning | 400-402 |
| Seed thử σ | 500-502 |
| Cell R1 | UIT-VSFC sentiment × XLM-R-base |
| Cell B1 | VSFC × ViSoBERT/PhoBERT |
| Cell B2 | VSMEC × XLM-R-base (fallback: `SetFit/ag_news` nếu không kịp) |
| Cell B4 | AG News × XLM-R-base |
| Shard | split by seed: `--shard 0/2` → seed chẵn, `--shard 1/2` → seed lẻ |

## Reference baselines (2026-10-16)

| Cấu hình | LR | n | Test F1 | Cal F1 | Test NLL |
|---|---|---|---|---|---|
| Head-only (sàn) | 2e-4 | 8 | 0.5879 ± 0.0043 | 0.5825 | 0.4228 |
| **Full FT (trần)** | **2e-5** | 5 | **0.8486 ± 0.0084** | 0.8165 | 0.2782 |
| Full FT (LR sai → collapse) | 2e-4 | 5 | 0.2199 ± 0.0000 | — | — |

**Ghi chú quan trọng:** LR 2e-4 (dùng cho LoRA rank 8) làm full FT collapse về majority class (F1 = 0.22). Full FT cần LR = 2e-5. Đây là bằng chứng LR phải tune riêng cho từng phương pháp PEFT — khớp với lit review (Biderman 2024, LR Matters 2026).

**Phát hiện phụ:** LoRA rank 32 (smoke test v2, 1 seed, 3 epoch) đạt Test F1 = 0.8442, NLL = 0.2064 — gần bằng full FT về phân biệt và tốt hơn về hiệu chuẩn (0.2064 vs 0.2782). Cần xác nhận ở Phase 1a.

## Near-duplicate QC (2026-10-16)

Phương pháp: TF-IDF `char_wb` (3-5 gram), cosine similarity với train. Ngưỡng 0.90, 0.95, 0.98.

| Split | ≥0.90 | ≥0.95 | ≥0.98 | max_sim_mean |
|---|---|---|---|---|
| Calibration | 12.20% | 5.19% | 0.69% | 0.652 |
| Test | 10.44% | 3.81% | 0.56% | 0.649 |

**Ghi chú:** Ghi chú cũ trong CHANGELOG (test ~2.6% ở ngưỡng 0.95) dùng TF-IDF word-level; phương pháp mới dùng char_wb n-gram nên bắt được nhiều cặp hơn. Cả hai hợp lệ, khác định nghĩa "gần trùng". Không đủ để đổi kết luận — leak ảnh hưởng cả 5 rank như nhau.

Script: `code/check_near_dup.py`. Kết quả thô: `results/analysis/near_dup_vsfc.json`.
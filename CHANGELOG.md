# Changelog

## 2026-10-17 — Cycle 3b (Patch `run_pilot.py` cho LR per rank)

### Thay đổi

`code/run_pilot.py` thêm `--lr_map <path>` (JSON `{rank: lr}` hoặc `{rank: {"best_lr": lr}}`). Khi được cung cấp, mỗi rank dùng LR từ map thay vì `--lr`/`--lr_list`. Khi không cung cấp, hành vi cũ giữ nguyên (cross product ranks × lr_list).

### Lý do

Phase 1 cần chạy rank {1, 4, 8, 16, 32} với **LR khác nhau theo rank** (từ `tuning_decision.json`). Trước patch, `run_pilot.py` chỉ có `--lr` (1 giá trị) và `--lr_list` (cross product với ranks). Muốn chạy 1 lệnh duy nhất cho cả 5 rank cần thêm `--lr_map`.

### Vị trí patch

- Dòng ~386: thêm `ap.add_argument("--lr_map", ...)` sau `--lr_list`.
- Hàm `build_jobs`: đọc `args.lr_map` (nếu có), gán `lr_map = {int(k): float(v["best_lr"])}`; set `lr_list = [None]` để vòng lặp ngoài chạy 1 lần.
- Hàm `add()` inner: nếu `lr_map` có → tra `lr_map.get(rank)`; rank không có trong map → skip (không thêm job).

### Test (local, 2026-10-17)

- `py_compile` pass; `--help` hiển thị `--lr_map`.
- Test với `_inbox/_test_decision.json` (5 rank từ seed 400): 5 jobs, mỗi rank có LR đúng.
- Test backward compat (`lr_map=None`): 4 jobs cross product như cũ.

### Ảnh hưởng

- **Không ảnh hưởng** các lệnh tune cũ (không truyền `--lr_map` → hành vi giữ nguyên).
- **Không ảnh hưởng** seed 400 cũ (chạy trước patch, `run_id` không đổi vì `run_pilot.py` không đổi giữa `6ef174b` và commit này ở đoạn tính `run_id`).
- R1 (seed 200-215) sẽ dùng `--lr_map tuning_decision.json`.

## 2026-10-17 — Cycle 3 (Chốt trước Phase 1)

### Kế hoạch phân tích v2.0 — mục 9 (Sai lệch)

Bốn dòng mới trong `docs/analysis_plan_v2.0.md` mục 9, ghi **trước khi đọc kết quả seed 401–402 và trước R1**:

1. **Quy tắc mở rộng lưới LR** đúng một bước ngoài biên (5e-5 ở biên dưới, 2e-3 ở biên trên) cho rank có LR chọn (sau quy tắc hòa) chạm biên lưới sau 3 seed tune. Không mở rộng quá một bước.
2. **Độ nhạy H1**: (i) TOST trực tiếp hiệu số trung bình rank 32 trừ rank 1, δ = 0,01, CI 90%; (ii) kiểm lệch tuyến tính `log2(rank)` so với rank dạng phân loại. H1 đã đăng ký vẫn là phân tích chính.
3. **Định nghĩa lượt "sập"**: Macro-F1 calibration < 0,25 (dự đoán lớp đa số, F1 ≈ 0,2207). Lượt sập **giữ** trong bảng tuning và tính trung bình chọn LR; báo cáo tỷ lệ sập theo (rank, LR).
4. **Quy trình ghép shard tuning**: seed 400 chỉ đặt ở `tuning_gpu0.csv`; dedupe theo `run_id`; assert 60 dòng trước khi chạy analyze_tuning.

### Sửa lỗi trong kế hoạch sáng 10-17

- Cell 1 trong hướng dẫn cũ copy seed 400 vào cả 2 file → 80 dòng. Sửa: **chỉ gpu0**; merge thêm `drop_duplicates("run_id")` + assert 60.
- Dòng log kỳ vọng đúng là `<N> jobs in this shard, <X> already done, <Y> to run`, không phải `[run_pilot] skip`.

### Ghi chú nhất quán giữa các commit

`run_pilot.py` **không đổi** giữa commit `6ef174b` (chạy seed 400) và `dd8be95` (hiện tại) — kiểm bằng `git diff --stat 6ef174b dd8be95 -- code`. Vì vậy 20 lượt seed 400 cũ và 40 lượt seed 401–402 mới chạy cùng mã huấn luyện. Cột `git` trong CSV hiển thị hai hash khác nhau — đây là metadata, không phải điều kiện không nhất quán.

### Tài liệu mới

- `docs/exploratory_results.md` mục 10.2: bảng lưới LR seed 400 (5 rank × 4 LR), tối ưu chạm biên ở rank 4 (trên), rank 8/16/32 (dưới).
- `_inbox/kaggle_cells_1017.py`: các cell Kaggle sáng 10-17 đã sửa (không commit).

### Còn mở

- OSF: cân nhắc cập nhật/addendum cho 4 dòng mục 9 (theo quy trình OSF hiện hành).
- Xác nhận TS. Huyên về quy tắc mở rộng lưới (mục 9, dòng 1).

## 2026-10-15 — Cycle 2b (Đổi hướng đề tài và thống nhất tài liệu)

### Quyết định
Đề tài chuyển từ "capacity need của lớp/module có đo được, ổn định, chuyển giao được không?" sang "rank LoRA toàn cục ảnh hưởng gì đến Macro-F1 và NLL, và có phụ thuộc ngân sách huấn luyện (3 so với 6 epoch) không?". Đây là thay đổi ngoài kế hoạch gốc, không phải nhánh dự phòng. Lý do, so sánh chi tiết, ánh xạ giả thuyết và việc còn mở: xem `docs/pivot_v2.md`. Trạng thái: đề xuất nội bộ.

### Căn cứ
- Pilot C1 (125 lượt): G1 NO-GO ở mức lớp (power 4,6%).
- Nhóm lớp (ICC 0,03) và loại module (ICC tối đa 0,26) âm tính với 5 seed.
- Stage 1a (ρ = 0,88, 2 seed) không lặp lại ở Stage 1c (ρ = 0,008, 5 seed); gap giảm ~6× ở 6 epoch.
- Gộp khám phá: F1 −0,0013 [−0,0051; +0,0025], NLL +0,0071 [+0,0035; +0,0107].

### Thống nhất tài liệu
- `README.md`: viết lại theo hướng mới; loss ghi đúng (CE không trọng số); pilot ghi đúng 3 epoch; thêm cấu trúc `docs/`.
- `docs/analysis_plan.md`: bf16 ghi đúng (T4 không có bf16 gốc); thêm điều kiện nhãn topic cho R2; tách nguồn đánh giá (test so với calibration); ghi chú trùng nhãn H1–H3 với bản 4.
- `docs/roadmap_v2.md`: 12 seed/96 lượt đổi thành 16 seed/128 lượt (khớp analysis_plan), δ 0,005 đổi thành 0,01, tính lại chi phí (~32 GPU-giờ, 170 lượt); đánh dấu các bước 0.1, 0.5, 0.6 đã xong.
- `docs/decisions.md`: sửa loss; ghi test là chỉ số chính của họ xác nhận; ghi kết quả Stage 1 và hướng hiện hành; Stage 1d chưa chạy.
- `docs/workflow.md`: `.sh` đổi thành `.ps1` theo `scripts/`.
- `code/analyze_global.py`: mặc định `--delta_f1` từ 0,005 thành 0,01. Không đổi logic.
- Thêm `docs/pivot_v2.md`.

### Lưu trữ để không mất chi tiết hướng cũ
- `docs/archive/proposal_v4_capacity_need.md`, `docs/archive/spec_v2_pilot_and_grid.md`: bản markdown của hai file Word (công thức giữ nguyên), `docs/archive/README_pre_pivot.md`: README cũ.
- `docs/exploratory_results.md`: số liệu và chú thích pilot, Stage 1a–1c; ghi rõ các câu nháp quá mạnh trong CHANGELOG không được dùng trong bài.

### Việc còn mở
1. TS. Huyên duyệt; commit và `git tag analysis-plan-v1` trước khi chạy Giai đoạn 1.
2. Kiểm tra precision trên T4 (`get_device_capability`).
3. Nhãn topic cho R2 (file thô chỉ có sentiment) hoặc chuyển sang AG News.
4. Tên đề tài (mục 6 của `pivot_v2.md`).


## 2026-10-15 — Cycle 2a (Smoke test v2 — PASS)

### Mục đích
Xác nhận `run_pilot.py` v2 chạy đúng trên Kaggle T4 với cấu hình global sweep.

### Setup
- Cell: SMOKE
- Config: `--sweep_global --ranks 1 32 --epochs_list 3 --seeds 900`
- Model: FacebookAI/xlm-roberta-base
- Precision: **bf16** (native T4)
- Batch 32, LR 2e-4, max_len 128
- Eval file: test.csv (chưa đổi)

### Env thực tế Kaggle
- torch 2.11.0+cu128
- transformers 5.16.1
- peft 0.20.0
- GPU: Tesla T4

### Kết quả

| Chỉ số | Rank 1 | Rank 32 | Ratio |
|---|---|---|---|
| Test Macro-F1 | 0.8141 | 0.8442 | — |
| Test NLL | 0.2408 | 0.2064 | — |
| Cal Macro-F1 | 0.7742 | 0.7963 | — |
| Cal NLL | 0.2181 | 0.2024 | — |
| LoRA params | 165,888 | 5,308,416 | **32.0×** |
| Seconds | 398 | 468 | — |

### Artifacts (đã verify)
- `smoke.csv` (2 rows)
- `epochwise/*.json` (3 epochs/run)
- `logits/*.npz` (eval + cal logits + gold)
- `meta.jsonl` (env info)

### Kết luận
- ✅ Code v2 hoạt động đúng
- ✅ Logits + epochwise logging đầy đủ
- ✅ Param ratio 32.0 chính xác
- ✅ Thời gian ~400s/run khớp ước tính

### Cần chốt trước global sweep
- Đổi `--eval_file calibration.csv` cho primary (đồng bộ với Stage 1c)
- Test.csv chỉ báo cáo cuối
- Ghi vào `docs/analysis_plan.md`

**Giải quyết (2026-10-15, Cycle 2b):** kế hoạch phân tích chọn **test** làm chỉ số chính của họ xác nhận (LR đã khóa bằng calibration, không có gì được chọn bằng test); calibration chỉ để fit nhiệt độ. Ghi chú "Precision: bf16 (native T4)" ở trên là sai: T4 không có bf16 gốc, `meta.jsonl` chỉ cho biết `bf16` trên Tesla T4 (có thể mô phỏng). Còn mở, xem `docs/pivot_v2.md` mục 8.

## 2026-10-15 — Cycle 1h (Stage 1c — full dose-response, 6 modules × 4 ranks × 5 seeds)

### Kết quả

**Spearman rho (dose-response):** Không module nào significant (Holm m=6, tất cả p_holm = 1.000).

**Paired t-test rank 32 vs rank 1:** Không module nào significant (Holm m=6, tất cả p_holm = 1.000).

**Signal `intermediate.dense` từ Stage 1a (rho=0.878) biến mất ở 5 seeds (rho=0.008).**
→ Stage 1a là fluke, không phải signal thật.

**Best rank per module rải rác:** query=4, key=1, value=4, attn_out=16, inter_ffn=4, out_ffn=32.
→ Pattern ngẫu nhiên, không có dose-response.

### Kết luận

**Negative result ở cả 3 resolution:**
- Single-layer (12): ρ = 0.0
- Layer group (3): ICC = 0.03
- Module type (6): rho n.s., delta n.s.

**Capacity need không đo được ở bất kỳ resolution nào với training budget thực tế.**

### Hàm ý cho paper

"Score dùng trong AdaLoRA/BaRA/IGU-LoRA có thể đang tối ưu trên noise."

Paper narrative: "Is Capacity Need a Measurable Module Property? A Negative Result."

### File
- `results/pilot/stage1c_merged.csv` (120 rows)
- `results/analysis/G1_stage1c.json`

### Next
- Power analysis cho Stage 1c
- TOST equivalence test (δ = 0.01)
- Quyết định: viết paper hay chạy Stage 2

## 2026-10-14 — Cycle 1g (Stage 1c Session 1 — ranks 1,16)

### Kết quả
- 60 jobs, 6 modules × 2 ranks × 5 seeds × 6 epochs
- File: `results/pilot/stage1c_s1_results.csv`

### Paired delta (rank16 - rank1):

| Module | Mean | Std |
|---|---|---|
| attention.output.dense | +0.0070 | 0.0044 |
| output.dense | +0.0007 | 0.0075 |
| attention.self.query | -0.0009 | 0.0085 |
| attention.self.key | -0.0001 | 0.0135 |
| attention.self.value | -0.0095 | 0.0095 |
| intermediate.dense | +0.0041 | 0.0186 |

### Nhận xét
- `intermediate.dense` mất tín hiệu ở rank 16 (delta +0.004 vs +0.020 ở rank 32, Stage 1a)
- `attention.output.dense` xuất hiện tín hiệu ổn định (tất cả 5 seeds dương)
- `attention.self.value` có delta âm (rank 16 tổn thương)
- → Ủng hộ giả thuyết underfitting artifact cho `intermediate.dense`

### Next
- Chạy Session 2 (ranks 4, 32) để hoàn thành dose-response

## 2026-10-05 — Cycle 1f (Stage 1b — 6-epoch test)

### Kết quả

intermediate.dense, 3 vs 6 epochs:

| Seed | Rank | 3-ep F1 | 6-ep F1 | 3-ep Δ | 6-ep Δ |
|---|---|---|---|---|---|
| 100 | 1 | 0.7956 | 0.8131 | — | — |
| 100 | 32 | 0.8129 | 0.8162 | +0.017 | +0.003 |
| 105 | 1 | 0.7877 | 0.8043 | — | — |
| 105 | 32 | 0.8101 | 0.8082 | +0.022 | +0.004 |
| **Mean** | | | | **+0.020** | **+0.0035** |

### Phát hiện

Gap giữa rank 1 và rank 32 **giảm 6×** khi tăng epochs 3 → 6.

Diễn giải: tín hiệu "capacity need" ở 3 epochs phần lớn là **underfitting artifact**. Với 6 epochs, rank 1 hội tụ gần bằng rank 32.

### Hàm ý cho paper

"Capacity need" đo được ở 3 epochs không phải thuộc tính module, mà là **artifact của training budget**.

Rank allocation dựa trên score đo ở 3 epochs có thể sai khi training budget thay đổi.

### NLL phụ
- 6-epoch NLL tệ hơn 3-epoch với hầu hết configs → có thể overfitting nhẹ.

### Next
- Stage 1c: 6 modules × 2 ranks × seeds {105-109} × **6 epochs** (48 jobs mới + 2 cũ)
- Kỳ vọng: delta 6-epoch ~+0.003, cần 10 seeds để đánh giá

## 2026-10-05 — Cycle 1e (Stage 1a — dose-response)

### Kết quả

**intermediate.dense (FFN) — có tín hiệu monotone:**

| Rank | seed 100 | seed 105 |
|---|---|---|
| 1 | 0.7956 | 0.7877 |
| 4 | 0.8022 | 0.8013 |
| 16 | 0.8078 | 0.8179 |
| 32 | 0.8129 | 0.8101 |

- Spearman rho = **+0.878**, p = 0.004
- Per-seed: {100: +1.0, 105: +0.8} — cả 2 seeds monotone
- Delta rank 32 vs rank 1: **+0.020**

**attention.self.key — chaotic:**

- rho = -0.293, p = 0.48
- Per-seed: {-0.4, -0.6} — không consistent
- → Loại, noise

### Kết luận

- Lần đầu tiên phát hiện signal monotone ở bất kỳ resolution nào.
- `intermediate.dense` (FFN) có capacity need thật.
- Giả thuyết "capacity need không tồn tại" bị bác bỏ ở module-type.

### Cảnh báo
- Chỉ 2 seeds. Effect 0.020 > τ ước lượng ban đầu (0.0026). Có thể là may mắn.

### Next
- Stage 1b: 6-epoch test cho intermediate.dense (~1.5h)
- Stage 1c: 10 seeds module-type (~7h)

## 2026-10-04 — Cycle 1c (Power analysis)

**Observed std** of (rank32 - rank1) delta = 0.01195.

| Effect size | n=5 | n=10 | n=20 | n=30 | n=50 |
|---|---|---|---|---|---|
| 0.005 | 0.11 | 0.22 | 0.43 | 0.60 | 0.83 |
| 0.010 | 0.30 | 0.66 | 0.94 | 0.99 | 1.00 |
| 0.015 | 0.57 | 0.94 | 1.00 | 1.00 | — |
| 0.020 | 0.80 | 1.00 | 1.00 | — | — |

**Kết luận:**
- 5 seeds chỉ detect được effect ≥ 0.020 (power ≥ 0.8)
- Effect quan sát được: 0.005–0.010 → thiếu power
- Cần 20 seeds cho effect 0.010, 50 seeds cho effect 0.005

**Hàm ý:** Negative result hiện tại chưa kết luận được. Cần chạy thêm seeds hoặc equivalence test với bound δ = 0.02.

**File:** `results/analysis/power_analysis.txt` (nếu lưu output), `code/power_analysis.py`

## 2026-10-04 — Cycle 1b (Module-type pilot)

### Setup
- 6 module types × 2 ranks × 5 seeds = 60 lượt
- Config: XLM-R base, batch 32, LR 2e-4, 3 epochs
- File: `results/pilot/results_type.csv`

### Kết quả ICC per module

| Module | ICC | rank1 | rank32 |
|---|---|---|---|
| intermediate.dense | **0.264** | -0.0067 | +0.0038 |
| attention.output.dense | 0.129 | -0.0027 | +0.0034 |
| output.dense | 0.111 | -0.0019 | +0.0037 |
| attention.self.key | 0.069 | +0.0042 | -0.0022 |
| attention.self.query | 0.061 | -0.0052 | +0.0010 |
| attention.self.value | 0.000 | -0.0053 | -0.0054 |

### ANOVA (attention vs FFN)
- Rank 1: F=0.192, p=0.665
- Rank 32: F=0.771, p=0.388

### Kết luận

Ở cả 3 resolution đều NO-GO:
- Single layer: ρ = 0.0
- Layer group: ICC = 0.03
- Module type: ICC max = 0.26

→ Capacity need không đo được ở bất kỳ resolution nào với 5 seeds.

**Tín hiệu yếu có tồn tại:**
- `intermediate.dense` (FFN): rank 32 tốt hơn rank 1 (+0.0105), ICC = 0.26
- `attention.self.value`: ICC = 0 — hoàn toàn không phản ứng với rank
- Attention nhìn chung không phản ứng với rank

### Next
- Chạy power analysis để chứng minh negative result không do thiếu power
- Viết paper audit negative result với equivalence test

## 2026-10-04 — Cycle 1b (Coarse-unit analysis)

### Phân tích coarse unit (3 layer groups: bottom/middle/top)

Chạy `code/analyze_coarse.py` trên pilot hiện có.

| Chỉ số | Rank 1 | Rank 32 |
|---|---|---|
| ICC | 0.027 | 0.037 |
| ANOVA p | 0.633 | 0.559 |
| Group mean ΔF1 | -0.005 đến +0.001 | -0.002 đến +0.002 |

### Phát hiện: seed effect chi phối

Per-seed group mean ΔF1 cho thấy:
- Seed 100: mọi group âm (-0.02)
- Seed 103: mọi group dương (+0.01)
- Các seed khác: gần 0

→ ΔF1 phụ thuộc SEED, không phụ thuộc LAYER/GROUP.

### Kết luận

Ở cả single-layer VÀ layer-group, capacity need KHÔNG đo được. Profile ΔF1 chủ yếu là seed noise.

### Next
- Chạy module-type pilot (attention vs FFN) để kiểm tra xem tín hiệu có ở mức module không
- Nếu không → accept negative result, viết paper "audit negative result"

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

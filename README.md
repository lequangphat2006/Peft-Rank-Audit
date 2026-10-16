<div align="center">

[![OSF Preregistration](https://img.shields.io/badge/OSF-Preregistered-blue)](https://doi.org/10.17605/OSF.IO/A2MN5)

# PEFT Rank Audit

**Kiểm toán ảnh hưởng của rank LoRA đến khả năng phân biệt và hiệu chuẩn**

*Rank LoRA toàn cục có cải thiện Macro-F1 không, có làm thay đổi NLL không, và hiệu ứng đó có phụ thuộc ngân sách huấn luyện không?*

**Pre-registration:** [10.17605/OSF.IO/A2MN5](https://doi.org/10.17605/OSF.IO/A2MN5) (public 2026-10-16)

</div>

---

> **Đề tài đã đổi hướng ngày 2026-10-15** sang câu hỏi rank toàn cục. Pre-registration OSF: [10.17605/OSF.IO/A2MN5](https://doi.org/10.17605/OSF.IO/A2MN5). TS. Huyên đã uỷ quyền toàn bộ quyết định cho tác giả chính; kế hoạch phân tích đã khóa bằng tag `analysis-plan-v2.0`.

## Giới thiệu

Các phương pháp rank allocation thích ứng (**AdaLoRA**, **IGU-LoRA**, **BaRA**, **StatLoRA**) cho rank khác nhau giữa các thành phần của mô hình, dựa trên giả định rằng mỗi thành phần có một "capacity need" riêng và đo được. Đề tài khởi đầu nhằm kiểm chứng giả định đó. Pilot với 5 seed không cho thấy tín hiệu vượt nhiễu seed (ở mức lớp, nhóm lớp và loại module), nên đề tài hiện lùi một bước và hỏi câu cơ bản hơn: **tăng rank LoRA toàn cục (cùng rank cho 12 lớp × 6 module) có ích không?**

**Đóng góp dự kiến:**

1. Một thiết kế đăng ký trước, đủ power, để đo ảnh hưởng của rank toàn cục lên Macro-F1 (kiểm định tương đương, TOST) và NLL, trên XLM-R-base với UIT-VSFC.
2. Kiểm tra hiệu ứng đó có đổi theo ngân sách huấn luyện (3 so với 6 epoch) không.
3. Báo cáo trung thực các kết quả âm tính và khám phá đã có (pilot, Stage 1a–1c), kèm power.

**Định vị:** Không đề xuất thuật toán mới. Giá trị nằm ở phép đo và kiểm toán. Phạm vi: một mô hình, tiếng Việt, LR tune theo rank, 3 và 6 epoch.

---

## Trạng thái

| Giai đoạn | Trạng thái | Ngày |
|---|---|---|
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
| Tune seed 400 (20 lượt) | ✅ | 2026-10-16 |
| Tune seed 401, 402 (chốt LR per rank) | ⏳ | 2026-10-17 |
| Giai đoạn 1: global sweep R1 (16 seed, 160 lượt) | ⏳ | — |
| Giai đoạn 2: cell B (B1, B2) | ⏳ | — |
| Bản thảo | ⏳ | — |

---

## Câu hỏi nghiên cứu (hiện hành)

Họ xác nhận H1–H3, hiệu chỉnh Holm (m = 3), α = 0,05. Chi tiết trong [`docs/analysis_plan_v2.0.md`](docs/analysis_plan_v2.0.md).

| Mã | Câu hỏi | Giả thuyết | Kiểm định |
|---|---|---|---|
| H1 | Rank toàn cục có cải thiện khả năng phân biệt không? | Hiệu ứng tổng rank 1→32 lên Macro-F1 tương đương 0 (6 epoch, LR per rank) | TOST, δ = 0,01 |
| H2 | Rank có ảnh hưởng đến NLL không? | NLL tăng theo rank (6 epoch) | Hệ số > 0, một phía |
| H3 | Hiệu ứng rank có phụ thuộc ngân sách huấn luyện không? | Độ dốc rank của F1 khác nhau giữa 3 và 6 epoch | F-test tương tác, hai nhánh |

**Ba điều kiện bền:** (i) H1, H2 giữ nguyên khi dùng LR tốt nhì; (ii) H3 hai nhánh (LR per rank chính + LR cố định 2e-4 kiểm tra); (iii) LR tune ở 6 epoch tái dùng cho 3 epoch.

Các giả thuyết H1–H7 của bản đề xuất 4 (capacity need của lớp, chuyển giao, proxy, mất cân bằng lớp, phân bổ oracle) đã được lưu trữ. Bảng ánh xạ ở [`docs/pivot_v2.md`](docs/pivot_v2.md) mục 4.

---

## Cấu trúc repo
peft-rank-audit/
├── code/ Source code
│ ├── rankcap.py Rank allocation + LoRA patterns + alpha_mode
│ ├── run_pilot.py Training loop + sweep (v3: --method, --alpha_mode, --data_fraction, ΔW stats, fp16 auto)
│ ├── analyze_global.py Phân tích H1-H3 (H3 hai nhánh, δ tương đối, forest plot)
│ ├── analyze_tuning.py Chọn LR theo rank (ngưỡng hòa 0.005)
│ ├── power_global_sweep.py Power và số seed cho global sweep
│ ├── check_near_dup.py QC near-duplicate
│ ├── run_phase1a.py Wrapper chạy R1 đợt 1 (LR per rank, 2 GPU)
│ ├── analyze_pilot.py Phân tích G1 (pilot, hướng cũ)
│ ├── analyze_.py Phân tích khám phá Stage 1a-1c (hướng cũ)
│ ├── prepare_vsfc.py Chuẩn bị dữ liệu
│ └── test_.py Unit test (29 tests)
├── data/ Dữ liệu (không push)
├── notebooks/ Notebook Kaggle
├── results/
│ ├── pilot/ Pilot C1, module-type, Stage 1a-1c (khám phá)
│ ├── smoke/ Smoke test v2
│ ├── preflight_ft/ Pre-flight LR sweep (chọn 2e-5 cho full FT)
│ ├── preflight_ft_full/ Pre-flight trên full data
│ ├── reference/ Trần (full FT) và sàn (head-only)
│ ├── tuning/ Tune LR theo rank (seed 400-402)
│ ├── global/ Global sweep R1 (sẽ có sau Giai đoạn 1)
│ └── analysis/ Báo cáo G1, near-dup QC, power, hình
├── scripts/ Script tự động
├── docs/
│ ├── pivot_v2.md Đổi hướng so với đề tài ban đầu (đọc đầu tiên)
│ ├── analysis_plan_v2.0.md Kế hoạch phân tích đăng ký trước (tag analysis-plan-v2.0)
│ ├── analysis_plan_v2.md Bản nháp cũ
│ ├── roadmap_v2.md Lộ trình, chi phí, checklist
│ ├── decisions.md Quyết định thiết kế + OSF DOI + toàn quyền
│ ├── limitations.md Tổng hợp hạn chế
│ ├── lit_review.md Rà soát tài liệu
│ ├── SETUP_KAGGLE.md Checklist setup Kaggle (torchao, fp16, multi-account)
│ ├── exploratory_results.md Kết quả khám phá (pilot, Stage 1a-1c, tune seed 400)
│ ├── workflow.md Quy trình Kaggle <-> local
│ └── archive/ Đề xuất bản 4, đặc tả bản 2, README cũ
├── README.md (file này)
└── CHANGELOG.md Nhật ký chi tiết

text

---

## Dữ liệu

**UIT-VSFC sentiment** — Vietnamese Students' Feedback Corpus

| Split | Số mẫu | Vai trò |
|---|---|---|
| Train | 12,780 | Huấn luyện |
| Calibration | 1,598 | Chọn LR; fit nhiệt độ (temperature scaling) |
| Test | 1,599 | Chỉ số chính của họ xác nhận H1-H3 (không dùng để chọn gì) |

**Phân bố lớp (train):**

| Nhãn | Ý nghĩa | Số mẫu | Tỷ lệ |
|---|---|---|---|
| 0 | Negative | 5,986 | 46.8% |
| 1 | Neutral | 522 | **4.1%** |
| 2 | Positive | 6,272 | 49.1% |

**Xử lý mất cân bằng:** Cross-entropy **không trọng số** (`--class_weight none`, mặc định của `run_pilot.py`). Không resampling.

**Near-duplicate leakage (TF-IDF char_wb 3-5 gram, cosine ≥ 0.95):**

- Calibration: 5.19%
- Test: 3.81%

(Ghi chú cũ trong CHANGELOG: test ~2.6% ở ngưỡng 0.95 dùng TF-IDF word-level; kết quả mới dùng char_wb n-gram nên bắt được nhiều cặp hơn.)

Data không push lên GitHub. Upload lên Kaggle Dataset `peft-data`.

---

## Môi trường

| Vai trò | Nền tảng | Cấu hình |
|---|---|---|
| Code + analyze | Local | Windows, torch CPU-only |
| Train | Kaggle | 2× T4 16GB, torch CUDA |

**Thư viện chính:** torch ≥ 2.1, transformers ≥ 4.45, peft 0.20.0, scikit-learn, scipy, statsmodels, pandas.

**Setup Kaggle bắt buộc:** xem [`docs/SETUP_KAGGLE.md`](docs/SETUP_KAGGLE.md). Quan trọng nhất:
- Gỡ `torchao` 0.10.0 trước khi chạy (xung đột peft 0.20)
- Dùng `--precision fp16` (T4 không có bf16 gốc, benchmark 3.27× nhanh hơn bf16 mô phỏng)

---

## Cách chạy

### Local — chạy test

```bash
python -m pytest code/test_run_pilot_v3.py code/test_run_pilot.py code/test_rankcap.py -v
Kaggle — setup và chạy
Xem docs/SETUP_KAGGLE.md để có checklist đầy đủ. Bản tóm tắt:

python
# Cell 0: Setup (gỡ torchao + clone code)
import subprocess, os, shutil, sys
subprocess.run(["pip","uninstall","-y","torchao"], capture_output=True)
os.chdir("/kaggle/working")
if os.path.exists("/kaggle/working/code"): shutil.rmtree("/kaggle/working/code")
subprocess.run(["git","clone","https://github.com/lequangphat2006/Peft-Rank-Audit.git",
                "/kaggle/working/code"], check=True)
sys.path.insert(0, "/kaggle/working/code/code")
Local — phân tích
bash
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
Nhật ký chạy
Cycle	Ngày	Session	Kết quả	Ghi chú
0	2026-10-03	Local setup	9/9 pytest pass	peft 0.21.2
0	2026-10-03	Push GitHub	commit 495f06d	18 files
0	2026-10-03	Kaggle setup	Dataset mounted	2× T4 16GB
1	2026-10-03	Smoke test	batch 8: 0.775, batch 32: 0.682	1 epoch
1	2026-10-03	LR grid	best LR = 2e-4 (F1 0.8255)	4 epochs
1	2026-10-04	Pilot + G1	125/125, ρ=0.0	NO-GO single-layer
1	2026-10-04	Nhóm lớp, loại module, power	ICC 0.03 / 0.26	Âm tính, 5 seed thiếu power
1	2026-10-05	Stage 1a, 1b	intermediate.dense ρ=0.88 (2 seed); gap giảm 6× ở 6 epoch	Nghi underfitting
1	2026-10-15	Stage 1c	Không module nào có ý nghĩa (Holm=1.0)	Tín hiệu 1a biến mất
2a	2026-10-15	Smoke test v2	PASS, ~400 s/lượt (3 epoch)	run_pilot.py v2
2b	2026-10-15	Đổi hướng đề tài	Rank toàn cục, docs/pivot_v2.md	
2c	2026-10-16	v3 code + fp16 benchmark	3.27× nhanh hơn bf16	run_pilot.py v3
2c	2026-10-16	Reference baselines	head 0.588, full FT LR 2e-5: 0.849	5 seed mỗi cái
2c	2026-10-16	Tune seed 400	20 lượt, LR khác nhau theo rank, 3 collapse	Khám phá
2c	2026-10-16	Near-dup QC	cal 5.19%, test 3.81% @ 0.95	char_wb
2c	2026-10-16	Pre-registration OSF public	DOI 10.17605/OSF.IO/A2MN5	Tag osf-preregistered-v2.0
Chi tiết: xem CHANGELOG.md.

Quyết định thiết kế
Chi tiết: docs/decisions.md.

Mục	Giá trị
Base model	FacebookAI/xlm-roberta-base
Num labels	3 (sentiment)
Max length	128
LoRA rank	Toàn cục {1, 4, 8, 16, 32}, cùng cho 12 lớp × 6 module
LoRA alpha	2r (alpha_pattern); độ nhạy: fixed:16, rsLoRA (mức C)
Target modules	6/layer (Q/K/V/O/FFN), 72 module
Loss	Cross-entropy không trọng số
Batch size	32
Epochs	3 và 6 (chính); 10 + 20%/50% data (mức L)
LR	Tune theo rank trên calibration, lưới {1e-4, 2e-4, 5e-4, 1e-3}, seed 400-402
LR full FT	2e-5 (khác LoRA; 2e-4 collapse)
Warmup / weight decay / clip	10% + cosine / 0.01 / 1.0
Precision	fp16 (T4 không có bf16 gốc; benchmark 3.27×)
Seed khám phá	100–114
Seed xác nhận R1	200–215 (16)
Seed cell B	300–307 (8)
Seed tune	400–402 (3)
Seed variance probe	500–502 (3)
Seed reference	600–607 (8 cho head, 5 cho full FT)
Chỉ số chính	Macro-F1 và NLL trên test
Chỉ số phụ	Accuracy, F1 từng lớp, NLL và ECE sau temperature scaling
Giới hạn
Chi tiết đầy đủ: docs/limitations.md.

Một mô hình (XLM-R-base), một ngôn ngữ (tiếng Việt), chỉ encoder, chỉ phân loại văn bản.

Lớp neutral chỉ ~4% train; Macro-F1 bị chi phối bởi lớp này. δ = 0.01 ≈ 1–2 mẫu neutral.

Near-dup 3.81% test (char_wb, ≥ 0.95). Giả định "leak ảnh hưởng mọi rank như nhau" chưa được kiểm.

LR tune một lần ở 6 epoch, tái dùng cho 3 epoch. LR chốt ở lưới {1e-4, 2e-4, 5e-4, 1e-3} — có thể tối ưu ngoài lưới (đặc biệt rank 4).

α = 2r làm ΔW tỉ lệ với rank; độ nhạy α ở mức C. Reviewer có thể hỏi "đo rank hay đo scaling".

Split cố định — nhiễu do chọn tập huấn luyện không nằm trong phương sai seed.

SD dư 0.0081 đo từ Stage 1c (calibration, module-level, bf16). Chưa đo trên test + fp16 + LR per rank + rank toàn cục.

Pre-registration nội bộ (OSF Preregistration, không phải Registered Report).

Tài liệu lưu trữ của hướng ban đầu
docs/archive/proposal_v4_capacity_need.md: đề xuất bản 4 (định nghĩa formal capacity need, bảy giả thuyết H1–H7, thiết kế C1–C7).

docs/archive/spec_v2_pilot_and_grid.md: đặc tả triển khai bản 2.

docs/archive/README_pre_pivot.md: README trước khi đổi hướng.

docs/exploratory_results.md: số liệu và chú thích của pilot và Stage 1a–1c.

Tài liệu tham khảo
Hu et al. (2022) — LoRA. ICLR.

Zhang et al. (2023) — AdaLoRA. ICLR.

Kalajdzievski (2023) — rsLoRA. arXiv:2312.03732.

Biderman et al. (2024) — LoRA Learns Less and Forgets Less. TMLR.

Aghajanyan et al. (2021) — Intrinsic dimensionality. ACL.

Nguyen & Nguyen (2020) — PhoBERT. EMNLP Findings.

Conneau et al. (2020) — XLM-R. ACL.

Rà soát đầy đủ: docs/lit_review.md.

Liên hệ
Tác giả liên hệ: TS. Đinh Thị Hồng Huyên

Đồng tác giả: Lê Quang Phát

Đơn vị: Khoa Công nghệ Thông tin, Trường Đại học Quy Nhơn

<div align="center"> <sub>Last updated: 2026-10-16 · Status: pre-registration public (OSF DOI 10.17605/OSF.IO/A2MN5), Phase 1 tune in progress</sub> </div> ```
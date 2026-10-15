<div align="center">

# PEFT Rank Audit

**Kiểm toán ảnh hưởng của rank LoRA đến khả năng phân biệt và hiệu chuẩn**

*Rank LoRA toàn cục có cải thiện Macro-F1 không, có làm thay đổi NLL không, và hiệu ứng đó có phụ thuộc ngân sách huấn luyện không?*

</div>

---

> **Đề tài đã đổi hướng ngày 2026-10-15.** Câu hỏi ban đầu ("capacity need của một lớp/module có đo được, ổn định và chuyển giao được không?") cho kết quả âm tính ở mọi mức phân giải đã thử. Xem [`docs/pivot_v2.md`](docs/pivot_v2.md) để biết đổi gì, vì sao, và những việc còn mở. Hướng mới chưa được TS. Huyên duyệt.

## Giới thiệu

Các phương pháp rank allocation thích ứng (**AdaLoRA**, **IGU-LoRA**, **BaRA**, **StatLoRA**) cho rank khác nhau giữa các thành phần của mô hình, dựa trên giả định rằng mỗi thành phần có một "capacity need" riêng và đo được. Đề tài khởi đầu nhằm kiểm chứng giả định đó. Pilot với 5 seed không cho thấy tín hiệu vượt nhiễu seed (ở mức lớp, nhóm lớp và loại module), nên đề tài hiện lùi một bước và hỏi câu cơ bản hơn: **tăng rank LoRA toàn cục (cùng rank cho 12 lớp × 6 module) có ích không?**

**Đóng góp dự kiến:**

1. Một thiết kế đăng ký trước, đủ power, để đo ảnh hưởng của rank toàn cục lên Macro-F1 (kiểm định tương đương, TOST) và NLL, trên XLM-R-base với UIT-VSFC.
2. Kiểm tra hiệu ứng đó có đổi theo ngân sách huấn luyện (3 so với 6 epoch) không.
3. Báo cáo trung thực các kết quả âm tính và khám phá đã có (pilot, Stage 1a–1c), kèm power.

**Định vị:** Không đề xuất thuật toán mới. Giá trị nằm ở phép đo và kiểm toán. Phạm vi: một mô hình, tiếng Việt, LR cố định, 3 và 6 epoch.

---

## Trạng thái

| Giai đoạn | Trạng thái | Ngày |
|---|---|---|
| Code viết xong | ✅ | 2026-10-03 |
| Pytest 9/9 pass | ✅ | 2026-10-03 |
| Data prepared | ✅ | 2026-10-03 |
| Push GitHub | ✅ | 2026-10-03 |
| Kaggle setup | ✅ | 2026-10-03 |
| Smoke test | ✅ | 2026-10-03 |
| LR grid | ✅ | 2026-10-03 |
| Pilot C1 (125 lượt) | ✅ | 2026-10-04 |
| Phân tích G1 | ✅ NO-GO | 2026-10-04 |
| Stage 1a–1c (dose-response, 3 so với 6 epoch, loại module) | ✅ Âm tính | 2026-10-15 |
| Đổi hướng sang rank toàn cục (`docs/pivot_v2.md`) | ✅ Đề xuất nội bộ | 2026-10-15 |
| Smoke test v2 (`run_pilot.py` v2, Kaggle T4) | ✅ | 2026-10-15 |
| Kế hoạch phân tích đăng ký trước (`docs/analysis_plan.md`) | ⏳ Bản nháp, chờ TS. Huyên duyệt và tag `analysis-plan-v1` | 2026-10-15 |
| Giai đoạn 1: global sweep R1 (16 seed, 128 lượt) | ⏳ | — |
| Giai đoạn 2: nhân bản R2 (8 seed, 32 lượt) | ⏳ | — |
| Bản thảo | ⏳ | — |

---

## Câu hỏi nghiên cứu (hiện hành)

Họ xác nhận H1–H3, hiệu chỉnh Holm (m = 3), α = 0,05. Chi tiết trong [`docs/analysis_plan.md`](docs/analysis_plan.md).

| Mã | Câu hỏi | Giả thuyết | Kiểm định |
|---|---|---|---|
| H1 | Rank toàn cục có cải thiện khả năng phân biệt không? | Hiệu ứng tổng rank 1→32 lên Macro-F1 tương đương 0 (6 epoch) | TOST, δ = 0,01 |
| H2 | Rank có ảnh hưởng đến NLL không? | NLL tăng theo rank (6 epoch) | Hệ số > 0, một phía |
| H3 | Hiệu ứng rank có phụ thuộc ngân sách huấn luyện không? | Độ dốc rank của F1 khác nhau giữa 3 và 6 epoch | F-test tương tác |

Các giả thuyết H1–H7 của bản đề xuất 4 (capacity need của lớp, chuyển giao, proxy, mất cân bằng lớp, phân bổ oracle) đã được lưu trữ, không còn là giả thuyết xác nhận. Bảng ánh xạ ở [`docs/pivot_v2.md`](docs/pivot_v2.md) mục 4; nội dung và kiểm định đầy đủ của H1–H7 (bản 4) ở [`docs/archive/proposal_v4_capacity_need.md`](docs/archive/proposal_v4_capacity_need.md) mục 4. Nhãn H1–H3 ở trên **không** trùng nghĩa với H1–H3 của bản 4.

---

## Cấu trúc repo

```
peft-rank-audit/
├── code/                      Source code
│   ├── rankcap.py             Rank allocation + LoRA patterns
│   ├── run_pilot.py           Training loop + sweep (v2: --sweep_global, --shard, logits)
│   ├── analyze_global.py      Phân tích họ H1-H3 (hướng hiện hành)
│   ├── power_global_sweep.py  Power và số seed cho global sweep
│   ├── analyze_pilot.py       Phân tích G1 (pilot, hướng cũ)
│   ├── analyze_*.py           Phân tích khám phá Stage 1a-1c (hướng cũ)
│   ├── prepare_vsfc.py        Chuẩn bị dữ liệu
│   └── test_*.py              Unit test
├── data/                      Dữ liệu (không push)
├── notebooks/                 Notebook Kaggle
├── results/
│   ├── pilot/                 Pilot C1, module-type, Stage 1a-1c (khám phá)
│   ├── smoke/                 Smoke test v2
│   ├── global/                Global sweep R1/R2 (sẽ có sau Giai đoạn 1)
│   └── analysis/              Báo cáo G1, Stage 1c, power, hình
├── scripts/                   Script tự động
├── docs/
│   ├── pivot_v2.md            Đổi hướng so với đề tài ban đầu (đọc đầu tiên)
│   ├── analysis_plan.md       Kế hoạch phân tích đăng ký trước (bản nháp)
│   ├── roadmap_v2.md          Lộ trình, chi phí, checklist
│   ├── decisions.md           Quyết định thiết kế
│   ├── exploratory_results.md Kết quả khám phá của hướng cũ (pilot, Stage 1a-1c)
│   ├── workflow.md            Quy trình Kaggle <-> local
│   └── archive/               Đề xuất bản 4, đặc tả bản 2, README cũ (lưu trữ)
├── README.md                  (file này)
└── CHANGELOG.md               Nhật ký chi tiết
```

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

**Xử lý mất cân bằng:** Cross-entropy **không trọng số** (`--class_weight none`, mặc định của `run_pilot.py`; mọi kết quả đã có đều dùng cách này). Không resampling. Cờ `--class_weight balanced` có sẵn nhưng không dùng trong kế hoạch.

**Near-duplicate leakage (cosine ≥ 0.95):**

- 3.5% val có near-duplicate trong train
- 2.6% test có near-duplicate trong train

**Gate G0** (hướng cũ, thao túng mất cân bằng): N = 1500 khả thi cho ρ ∈ {1, 5, 20}. Hướng hiện hành không thao túng mất cân bằng.

Data không push lên GitHub. Upload lên Kaggle Dataset `peft-data`.

---

## Môi trường

| Vai trò | Nền tảng | Cấu hình |
|---|---|---|
| Code + analyze | Local | Windows, RTX 3050 4GB, torch CPU-only |
| Train | Kaggle | T4 16GB, torch CUDA |

**Thư viện chính:** torch ≥ 2.1, transformers ≥ 4.45, peft ≥ 0.12, scikit-learn, scipy, statsmodels, pandas.

Cài đặt:

```bash
pip install -r requirements.txt
```

---

## Cách chạy

### Local — chuẩn bị dữ liệu

```bash
python code/prepare_vsfc.py \
  --train data/raw/train.csv \
  --val   data/raw/val.csv \
  --test  data/raw/test.csv \
  --out_dir data/vsfc_sent \
  --text_col text --label_col label
```

### Local — chạy test

```bash
python -m pytest code/test_rankcap.py -v
```

### Kaggle — clone code

```bash
git clone https://github.com/lequangphat2006/Peft-Rank-Audit.git /kaggle/working/code
```

### Kaggle — pilot C1 (hướng cũ, đã chạy xong; giữ để tái lập)

Pilot chạy **3 epoch** (không phải 6), đánh giá trên calibration:

```bash
cd /kaggle/working/code
python run_pilot.py \
  --data_dir /kaggle/working/data/vsfc_sent \
  --model FacebookAI/xlm-roberta-base \
  --cell C1 \
  --sweep \
  --seeds 100 101 102 103 104 \
  --ranks 1 32 \
  --epochs 3 \
  --lr 2e-4 \
  --batch_size 32 \
  --max_len 128 \
  --eval_file calibration.csv \
  --out /kaggle/working/results_C1.csv
```

Lưu ý: `run_pilot.py` hiện là bản v2 (đánh giá cả test và calibration, thêm `--sweep_global`, `--shard`). Lệnh trên là lệnh lúc chạy pilot; bản v1 xem lịch sử git.

### Kaggle — global sweep (hướng hiện hành, chưa chạy)

Lệnh đầy đủ (hai tiến trình GPU, chạy lại để tiếp tục) nằm trong [`docs/roadmap_v2.md`](docs/roadmap_v2.md) mục 4. Chỉ chạy sau khi `analysis_plan.md` đã được khóa.

### Local — phân tích kết quả

```bash
# hướng cũ (pilot)
python code/analyze_pilot.py results/pilot/results_C1_merged.csv --cell C1
# hướng hiện hành (sau khi có results/global/global_merged.csv)
python code/analyze_global.py results/global/global_merged.csv --primary_epochs 6 --delta_f1 0.01 \
  --logits_dir results/global/logits --out results/analysis/global_v1.json
```

---

## Nhật ký chạy

| Cycle | Ngày | Session | Kết quả | Ghi chú |
|---|---|---|---|---|
| 0 | 2026-10-03 | Local setup | 9/9 pytest pass | peft 0.21.2 |
| 0 | 2026-10-03 | Push GitHub | commit 495f06d | 18 files |
| 0 | 2026-10-03 | Kaggle setup | Dataset mounted | 2× T4 16GB |
| 1 | 2026-10-03 | Smoke test | batch 8: 0.775, batch 32: 0.682 | 1 epoch |
| 1 | 2026-10-03 | LR grid | best LR = 2e-4 (F1 0.8255) | 4 epochs |
| 1 | 2026-10-04 | Pilot + G1 | ✅ 125/125, ρ=0.0 | NO-GO single-layer |
| 1 | 2026-10-04 | Nhóm lớp, loại module, power | ICC 0.03 / 0.26 | Âm tính, 5 seed thiếu power |
| 1 | 2026-10-05 | Stage 1a, 1b | `intermediate.dense` ρ=0.88 (2 seed); gap giảm 6× ở 6 epoch | Nghi underfitting |
| 1 | 2026-10-15 | Stage 1c | Không module nào có ý nghĩa (Holm=1.0) | Tín hiệu 1a biến mất |
| 2a | 2026-10-15 | Smoke test v2 | PASS, ~400 s/lượt (3 epoch) | `run_pilot.py` v2 |
| 2b | 2026-10-15 | Đổi hướng đề tài | Rank toàn cục, `docs/pivot_v2.md` | Chờ TS. Huyên duyệt |

Chi tiết: xem [CHANGELOG.md](CHANGELOG.md).

---

## Quyết định thiết kế

Chi tiết: [docs/decisions.md](docs/decisions.md).

| Mục | Giá trị |
|---|---|
| Base model | `FacebookAI/xlm-roberta-base` |
| Num labels | 3 (sentiment); R2 có thể khác |
| Max length | 128 |
| LoRA rank (hướng hiện hành) | Toàn cục {1, 4, 16, 32}, cùng cho 12 lớp × 6 module |
| LoRA base rank (pilot, LR grid) | 8 |
| LoRA alpha | 2r (`alpha_pattern`) |
| Target modules | 6/layer (Q/K/V/O/FFN), 72 module |
| Loss | Cross-entropy không trọng số |
| Batch size | 32 |
| Epochs | 3 và 6 (hai ngân sách, là biến của thí nghiệm). Pilot C1: 3 |
| LR | 2e-4 (chốt bằng grid ở rank 8) |
| Warmup / weight decay / clip | 10% + cosine / 0,01 / 1,0 |
| Precision | bf16 (T4 không có bf16 gốc, xem `docs/pivot_v2.md` mục 8) |
| Seed pilot và khám phá | 100–114 (không dùng lại để xác nhận) |
| Seed xác nhận R1 | 200–215 (16 seed) |
| Seed nhân bản R2 | 300–307 (8 seed) |
| Chỉ số chính | Macro-F1 và NLL trên test |
| Chỉ số phụ | Accuracy, F1 từng lớp, NLL và ECE sau temperature scaling |

---

## Giới hạn

- Một mô hình (XLM-R-base), một ngôn ngữ (tiếng Việt), chỉ encoder, chỉ phân loại văn bản.
- LR chốt ở rank 8 rồi cố định cho mọi rank; rank nhỏ có thể bị thiệt (có kiểm tra thăm dò 1b).
- Hai ngân sách huấn luyện (3 và 6 epoch); không kết luận về ngân sách khác.
- Split cố định, nên nhiễu do chọn tập huấn luyện không nằm trong phương sai seed.
- Lớp neutral chỉ ~4% train; 2,6% test có near-duplicate trong train.
- Kết quả âm tính về capacity need của pilot dựa trên 5 seed (power 4,6% ở mức lớp). Đó là "chưa phát hiện", không phải "không tồn tại".

---

## Tài liệu lưu trữ của hướng ban đầu

- [`docs/archive/proposal_v4_capacity_need.md`](docs/archive/proposal_v4_capacity_need.md): đề xuất bản 4 (định nghĩa formal capacity need, bảy giả thuyết H1–H7 và kiểm định, thiết kế C1–C7).
- [`docs/archive/spec_v2_pilot_and_grid.md`](docs/archive/spec_v2_pilot_and_grid.md): đặc tả triển khai bản 2 (cấu hình LoRA theo lớp, quy tắc G1, giao thức mất cân bằng).
- [`docs/archive/README_pre_pivot.md`](docs/archive/README_pre_pivot.md): README trước khi đổi hướng.
- [`docs/exploratory_results.md`](docs/exploratory_results.md): số liệu và chú thích quan trọng của pilot và Stage 1a–1c.

---

## Tài liệu tham khảo

- Hu et al. (2022) — LoRA. ICLR.
- Zhang et al. (2023) — AdaLoRA. ICLR.
- Aghajanyan et al. (2021) — Intrinsic dimensionality. ACL.
- Nguyen & Nguyen (2020) — PhoBERT. EMNLP Findings.
- Conneau et al. (2020) — XLM-R. ACL.

---

## Liên hệ

- **Tác giả liên hệ:** TS. Đinh Thị Hồng Huyên
- **Đồng tác giả:** Lê Quang Phát
- **Đơn vị:** Khoa Công nghệ Thông tin, Trường Đại học Quy Nhơn

<div align="center">
<sub>Last updated: 2026-10-15 · Status: đổi hướng sang rank toàn cục, chờ duyệt kế hoạch phân tích</sub>
</div>

<div align="center">

# PEFT Rank Audit

**Kiểm toán giả định nền tảng của rank allocation trong PEFT**

*Capacity need của một module có đo được, ổn định và chuyển giao được không?*

</div>

---

## Giới thiệu

Các phương pháp rank allocation thích ứng (**AdaLoRA**, **IGU-LoRA**, **BaRA**, **StatLoRA**) dựa trên một giả định nền tảng: **độ bất định của module là proxy cho capacity need của module đó**. Đề tài này kiểm chứng giả định đó một cách hệ thống.

**Ba đóng góp:**

1. Định nghĩa formal *capacity need* và đo lường bằng rank sweep.
2. Kiểm chứng proxy bằng thực nghiệm, so sánh uncertainty vs effective rank vs intrinsic dimension.
3. Lý thuyết về điều kiện hoạt động — khi nào proxy hợp lý, khi nào ngân sách tham số quan trọng hơn cách phân bổ.

**Định vị:** Không đề xuất thuật toán mới. Giá trị nằm ở phép đo và kiểm toán.

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
| Chạy chính | ⏳ | — |
| Bản thảo | ⏳ | — |

---

## Câu hỏi nghiên cứu

| Mã | Câu hỏi | Giả thuyết |
|---|---|---|
| RQ1a / H1 | Capacity need có đo được vượt nhiễu seed không? | Phương sai tương tác lớp × rank > 0 |
| RQ1b / H2 | Profile có ổn định qua seed không? | Split-half reliability ≥ ngưỡng |
| RQ1c / H3 | Tác động của các lớp có cộng tính không? | F1 thực ≈ F1 cộng tính |
| RQ1d / H4 | Profile có chuyển giao qua task/ngôn ngữ? | Kendall τb ≥ 50% trần tin cậy |
| RQ2a / H5 | Proxy nào dự đoán được profile? | τb > 0.3 cho ít nhất 1 proxy |
| RQ2b / H6 | Mất cân bằng lớp có làm giảm giá trị proxy? | τb giảm khi ρ tăng |
| RQ3 / H7 | Phân bổ theo oracle/proxy có vượt baseline? | Wilcoxon ghép cặp, TOST |

---

## Cấu trúc repo

```
peft-rank-audit/
├── code/                    Source code
│   ├── rankcap.py           Rank allocation + LoRA patterns
│   ├── run_pilot.py         Training loop + sweep
│   ├── analyze_pilot.py     Phân tích thống kê + G1
│   ├── prepare_vsfc.py      Chuẩn bị dữ liệu
│   └── test_rankcap.py      9 unit test
├── data/                    Dữ liệu (không push)
│   ├── raw/                 3 CSV gốc
│   └── vsfc_sent/           CSV đã chuẩn bị
├── notebooks/               Notebook Kaggle
├── results/                 Kết quả từ Kaggle
│   ├── pilot/               results_C1_*.csv
│   └── analysis/            G1 report, figures
├── scripts/                 Script tự động
├── docs/                    Workflow + decisions
├── README.md                (file này)
└── CHANGELOG.md             Nhật ký chi tiết
```

---

## Dữ liệu

**UIT-VSFC sentiment** — Vietnamese Students' Feedback Corpus

| Split | Số mẫu | Vai trò |
|---|---|---|
| Train | 12,780 | Huấn luyện |
| Calibration | 1,598 | Chọn LR, Temperature Scaling |
| Test | 1,599 | Đánh giá cuối |

**Phân bố lớp (train):**

| Nhãn | Ý nghĩa | Số mẫu | Tỷ lệ |
|---|---|---|---|
| 0 | Negative | 5,986 | 46.8% |
| 1 | Neutral | 522 | **4.1%** |
| 2 | Positive | 6,272 | 49.1% |

**Xử lý mất cân bằng:** Class-weighted Cross-Entropy. Không resampling.

**Near-duplicate leakage (cosine ≥ 0.95):**

- 3.5% val có near-duplicate trong train
- 2.6% test có near-duplicate trong train

**Gate G0:** N = 1500 khả thi cho ρ ∈ {1, 5, 20}.

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

### Kaggle — train pilot

```bash
cd /kaggle/working/code
python run_pilot.py \
  --data_dir /kaggle/working/data/vsfc_sent \
  --model FacebookAI/xlm-roberta-base \
  --cell C1 \
  --sweep \
  --seeds 100 101 102 103 104 \
  --ranks 1 32 \
  --epochs 6 \
  --lr 2e-4 \
  --batch_size 32 \
  --max_len 128 \
  --eval_file calibration.csv \
  --out /kaggle/working/results_C1.csv
```

### Local — phân tích kết quả

```bash
python code/analyze_pilot.py results/pilot/results_C1_merged.csv --cell C1
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

Chi tiết: xem [CHANGELOG.md](CHANGELOG.md).

---

## Quyết định thiết kế

Chi tiết: [docs/decisions.md](docs/decisions.md).

| Mục | Giá trị |
|---|---|
| Base model | `FacebookAI/xlm-roberta-base` |
| Num labels | 3 |
| Max length | 128 |
| LoRA base rank | 8 |
| LoRA alpha | 2r (`alpha_pattern`) |
| Target modules | 6/layer (Q/K/V/O/FFN) |
| Total modules | 72 |
| Batch size | 32 (T4) |
| Epochs | 6 (cố định) |
| LR | 2e-4 (chốt sau grid) |
| Warmup | 10% |
| Weight decay | 0.01 |
| Clip grad | 1.0 |
| Precision | bf16 |
| Seeds pilot | 100–104 |
| Ranks pilot | {1, 32} |

---

## Giới hạn

- Chỉ encoder-based, chỉ phân loại văn bản, hai ngôn ngữ (Việt và Anh).
- Capacity need có điều kiện theo ngữ cảnh (các lớp khác ở rank 8).
- Chỉ 12 lớp → kiểm định xếp hạng từng cell có power thấp.
- Mất cân bằng lớp được thao túng bằng cấp số nhân ở N nhỏ (12% train).
- Nhiễu seed không gồm biến thiên do chọn tập huấn luyện (split cố định).

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
<sub>Last updated: 2026-10-04 · Status: G1 NO-GO, switch to coarse units</sub>
</div>

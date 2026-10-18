# PEFT Rank Audit

**Kiểm toán ảnh hưởng của rank LoRA toàn cục đến khả năng phân biệt và hiệu chuẩn**

*Rank LoRA toàn cục có cải thiện Macro-F1 không, có làm thay đổi NLL không, và hiệu ứng đó có phụ thuộc ngân sách huấn luyện không?*

**Đăng ký trước (pre-registration):** [10.17605/OSF.IO/A2MN5](https://doi.org/10.17605/OSF.IO/A2MN5) (công khai 2026-10-16)

> **Trạng thái (2026-10-18):** Đã hoàn thành các thực nghiệm xác nhận (R1, B1, B2) và các phân tích hậu kiểm (V1–V8, độ nhạy α, AdaLoRA). Đang soạn bản thảo. Mọi phân tích được thiết kế sau khi đã thấy dữ liệu đều được gắn nhãn "hậu kiểm" hoặc "thăm dò" và ghi trong mục 9 của kế hoạch phân tích.

## Giới thiệu

Các phương pháp phân bổ rank thích ứng (AdaLoRA, IGU-LoRA, StatLoRA) cho các thành phần của mô hình rank khác nhau, dựa trên giả định rằng mỗi thành phần có một "nhu cầu dung lượng" riêng và đo được. Dự án này là một **kiểm toán đăng ký trước** để trả lời câu hỏi cơ bản hơn: *tăng rank LoRA toàn cục mang lại lợi ích gì, và cái giá là gì?*

**Đóng góp dự kiến:**

1. **Kiểm định tương đương (TOST) có đăng ký trước** cho ảnh hưởng của rank toàn cục lên Macro-F1 và NLL, với ngưỡng có ý nghĩa thực tế δ = 0,01 và quy tắc chọn LR theo rank được ghi trước.
2. **Chỉ ra các yếu tố có thể tạo ra "hiệu ứng rank" giả:** cách chọn learning rate (LR), độ chính xác fp16 và hệ số α = 2r. Mỗi yếu tố có bằng chứng ở mức khác nhau, xem phần Kết quả.
3. **Tính phụ thuộc mô hình:** kết luận trên XLM-R không chuyển sang PhoBERT.
4. **Báo cáo trung thực kết quả âm tính, độ bất định do mẫu test và các sai lệch so với đăng ký.**

AdaLoRA được chạy như một baseline **thăm dò**, chưa tune công bằng, nên không dùng để rút ra kết luận so sánh.

---

## Kết quả chính (tính đến 2026-10-18)

Thiết lập chính (R1): XLM-R-base, UIT-VSFC, 16 seed (200–215), 5 rank {1, 4, 8, 16, 32}, 3 và 6 epoch, LR tune theo rank bằng quy tắc đã đăng ký (160 lượt). Số liệu tổng hợp: `results/analysis/final_results.json`, `global_v1.json`, `v1b_h3_fixed.json`.

| Giả thuyết | Kết quả | Trạng thái |
|---|---|---|
| **H1** (tương đương F1) | Tổng hiệu ứng rank 1→32 trên Macro-F1: −0,003, CI 90% [−0,008; +0,002]. TOST đạt trong δ = 0,01 (p_holm = 0,016), tính theo phương sai giữa các seed. | Đạt theo seed; xem lưu ý về mẫu test bên dưới |
| **H2** (NLL) | NLL tăng theo rank: tổng +0,005 (khoảng 2%) ở 6 epoch, p_holm = 0,014. Nhỏ hơn ngưỡng thực tế δ_NLL = 0,01. | Có ý nghĩa thống kê, nhỏ về thực tế |
| **H3** (rank × epoch) | Nhánh chính (LR theo rank): tương tác +0,017, p thô = 0,0023, p_holm = 0,0070. Nhánh LR cố định 2e-4: tương tác −0,033 (4 rank, 4 seed). Hai nhánh **ngược dấu**. | Không bền vững theo quy tắc đã đăng ký |

**Lưu ý về mẫu test (V7).** Khoảng tin cậy của H1 ở trên chỉ phản ánh phương sai do seed. Khi bootstrap ghép cặp trên các mẫu test (1.599 mẫu, logits của 16 seed, 6 epoch), tổng hiệu ứng rank 1→32 vẫn là −0,003 nhưng CI 90% rộng hơn: [−0,013; +0,006], cận dưới vượt nhẹ ngưỡng −0,01. Kết quả gần như không đổi trên test đã loại near-duplicate (1.539 mẫu): −0,004, CI 90% [−0,013; +0,006]. Vì vậy, kết luận "tương đương trong δ = 0,01" đúng với biến thiên do huấn luyện, nhưng **chưa được xác lập khi tính thêm nhiễu do kích thước tập test**. F1 từng rank có CI 95% khoảng ±0,03, và F1 lớp neutral khoảng ±0,09 (≈ 67 mẫu test).

**Diễn giải H3.** Ở LR cố định, F1 tăng theo rank ở 3 epoch (tổng +0,049), trong khi ở nhánh LR theo rank, hai rank bị gán LR thấp nhất (8 và 32) kém ở 3 epoch. Vì LR được tune ở 6 epoch rồi dùng lại cho 3 epoch, một phần tương tác rank × epoch ở nhánh chính gắn với cách chọn LR chứ không chỉ với rank. Nhánh LR cố định mới có 4 seed và 4 rank, nên đây là bằng chứng mạnh nhưng chưa phải kết luận cuối cùng.

**Phụ thuộc mô hình (V4, B1, B2).**
- *PhoBERT (B1/V4, đã tune LR theo cùng quy tắc, 8 seed):* tăng rank từ 1 lên 32 cải thiện F1 khoảng +0,020 (từ 0,799 lên 0,819; CI 90% [+0,013; +0,028], TOST không đạt tương đương) và làm NLL tăng khoảng +0,021. Phần lớn lợi ích F1 đã đạt ở rank 16.
- *B2 (AG News, LR cố định 2e-4, 8 seed):* F1 +0,009 (CI 95% [+0,006; +0,012]), TOST không đạt.
- *XLM-R/UIT-VSFC (R1):* F1 tương đương theo seed (H1).

**Độ ổn định số học (V5, V5b).** Ở LR 1e-3, các cấu hình rank 16 và 32 sập (F1 < 0,25) cả ở fp32, nên đó không chỉ là lỗi của fp16. Ở LR 5e-4, rank 32 sập 2/3 seed khi dùng fp16 nhưng 0/3 khi dùng fp32 (seed 400–402), nên fp16 góp phần vào vùng LR ổn định của rank cao.

**Độ nhạy α (thăm dò, 3 seed).** Ở LR 2e-4, 6 epoch, hiệu NLL (rank 32 trừ rank 1) là dương với α = 2r (+0,029 / +0,003 / +0,035) và âm với α = fixed:16 (−0,020 / −0,0002 / −0,022) trên cả ba seed. Dấu nhất quán, nhưng với n = 3 đây chỉ là gợi ý rằng phần tăng NLL theo rank trong R1 liên quan đến hệ số α = 2r; chưa phải kết luận.

**AdaLoRA (thăm dò, chưa tune LR, 3 seed).** F1 khoảng 0,73 ở cấu hình đầu và khoảng 0,78 khi tắt orthogonal regularization, so với 0,834 của LoRA rank 8. Nguyên nhân khoảng cách còn lại chưa được chẩn đoán (lịch cắt tỉa, ngân sách rank cuối, LR), nên chưa kết luận AdaLoRA kém hơn LoRA toàn cục.

---

## Hạn chế đã biết

- Kết luận tương đương chỉ vững cho XLM-R-base trên UIT-VSFC và chỉ theo phương sai giữa các seed; B1 (PhoBERT) và B2 cho kết quả khác. Chưa có task hay mô hình thứ ba chạy đủ quy trình như R1, và chưa có decoder.
- Split train/calibration/test cố định, nên các khoảng tin cậy theo seed không phản ánh biến thiên do chia dữ liệu. Bootstrap trên mẫu test (V7) cho thấy CI của H1 rộng hơn khoảng tin cậy theo seed, xem phần trên.
- Phân tích LR cố định (V1, V1b), fp32 (V5, V5b), α và AdaLoRA là phân tích **hậu kiểm** hoặc **thăm dò**, thiết kế sau khi đã thấy dữ liệu R1.
- AdaLoRA chạy trước khi có đăng ký cho nó và dùng LR của rank 8, không phải LR đã tune cho AdaLoRA.
- Các cell B (B1, B2) ban đầu chạy với LR cố định 2e-4; B1 đã được tune lại ở V4, B2 thì chưa.

## Dữ liệu

UIT-VSFC sentiment (Vietnamese Students' Feedback Corpus). Dữ liệu **không** được phân phối trong repo này (`data/` bị bỏ qua trong `.gitignore`); hãy tải từ nguồn gốc và đặt vào `data/vsfc_sent/` (`train.csv`, `calibration.csv`, `test.csv`).

- Train: 12.780 | Calibration: 1.598 | Test: 1.599
- Near-duplicate với tập train (TF-IDF char_wb 3–5-gram, cosine ≥ 0,95): Calibration 5,19%, Test 3,75% (60 mẫu).

## Nhật ký chạy

| Chu kỳ | Ngày | Nội dung |
|---|---|---|
| 0–2 | 10-03 → 10-16 | Thiết lập, pilot, đổi hướng đề tài, đăng ký OSF |
| 3 | 10-17 | Tune LR (69 lượt), R1 xác nhận (160 lượt), AdaLoRA thăm dò |
| 4 | 10-18 | Hậu kiểm: V1/V1b (H3 theo LR cố định), V4 (PhoBERT), V5/V5b (fp32), V6 (AdaLoRA), V7 (bootstrap theo mẫu test), độ nhạy α |

## Cấu trúc repo

```text
peft-rank-audit/
├── code/
│   ├── run_pilot.py             Huấn luyện và sweep
│   ├── run_adalora.py           Baseline AdaLoRA (thăm dò)
│   ├── analyze_global.py        Phân tích H1-H3
│   ├── analyze_tuning.py        Chọn LR theo rank
│   ├── analyze_tost_b1.py       TOST cho B1 (PhoBERT)
│   ├── analyze_v7_true.py       Bootstrap trên mẫu test (từng rank)
│   ├── analyze_v7_drop_dup.py   Kiểm tra trên test đã loại near-duplicate
│   └── ...                      Các script phân tích khác
├── docs/
│   ├── analysis_plan_v2.0.md    Kế hoạch đăng ký trước (mục 9: sai lệch)
│   ├── limitations.md           Hạn chế
│   ├── exploratory_results.md   Nhật ký kết quả (R1, B1, B2, V1-V8)
│   └── ...
├── results/
│   ├── global/                  R1 (160 lượt) và logits (.npz)
│   ├── tuning/                  Tune LR (69 lượt)
│   ├── adalora/                 AdaLoRA thăm dò
│   ├── v4_phobert/              Tune LR cho PhoBERT
│   └── analysis/                JSON tổng hợp và biểu đồ
├── README.md
└── CHANGELOG.md
```

## Cách chạy

**1. Thiết lập môi trường (Kaggle):**

```bash
pip uninstall -y torchao
git clone https://github.com/lequangphat2006/Peft-Rank-Audit.git
cd Peft-Rank-Audit
pip install -r requirements.txt
cd code
```

**2. Chạy R1 (xác nhận):**

```bash
python run_pilot.py \
  --data_dir /kaggle/input/datasets/lequangphat260206/peft-data \
  --cell R1 --sweep_global --ranks 1 4 8 16 32 --epochs_list 3 6 \
  --seeds 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 \
  --lr_map results/tuning/tuning_decision.json --precision fp16
```

**3. Phân tích (chạy từ thư mục gốc của repo):**

```bash
python code/analyze_global.py results/global/global_merged.csv --delta_f1 0.01
python code/analyze_v7_true.py        # CI bootstrap theo mẫu test cho từng rank
python code/analyze_v7_drop_dup.py    # cần data/vsfc_sent/{train,test}.csv
python code/analyze_tost_b1.py        # TOST cho PhoBERT
```

Môi trường đã dùng: peft 0.21.2, transformers 5.16.1, torch 2.13 (xem `requirements.txt`).

## Tài liệu tham khảo

- Hu et al. (2022). LoRA. ICLR.
- Zhang et al. (2023). AdaLoRA. ICLR.
- Lee et al. (2026). Learning Rate Matters. arXiv:2602.04998.
- Biderman et al. (2024). LoRA Learns Less and Forgets Less. TMLR.

---

**Tác giả liên hệ:** TS. Đinh Thị Hồng Huyên
**Đồng tác giả:** Lê Quang Phát
**Đơn vị:** Khoa Công nghệ Thông tin, Trường Đại học Quy Nhơn

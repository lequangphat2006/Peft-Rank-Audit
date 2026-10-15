> **LƯU TRỮ — không còn hiệu lực.** Bản chuyển từ file Word `đặc tả triển khai bản 2 (pilot và lưới capacity need)` (tác giả Lê Quang Phát, 2026-10-03) sang markdown, giữ nguyên nội dung để không mất chi tiết của hướng ban đầu. Đề tài đã đổi hướng ngày 2026-10-15, xem `docs/pivot_v2.md`. Không sửa file này; nếu có công thức hoặc bảng bị lệch khi chuyển định dạng, đối chiếu với file Word gốc.

# Đặc tả triển khai (bản 2): pilot và lưới capacity need

Oct 3, 2026 · @Le Quang Phat

## 1. Nguyên tắc và thay đổi so với đặc tả cũ

Đặc tả này đi kèm bản đề xuất lan 4 và ưu tiên một điều: phép đo capacity need phải sạch (ghép cặp, cùng ngân sách, cùng hệ số co giãn) trước khi tối ưu hiệu năng. Nguyên tắc "performance-first" của đặc tả cũ chỉ áp dụng cho phần baseline, không cho sweep.

| Hạng mục          | Đặc tả cũ                                          | Đặc tả mới                                                                                                                |
|-------------------|----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| Đơn vị rank       | 12 lớp trong sweep, 48 module trong Dynamic        | 12 lớp; rồi sweep thêm theo loại module                                                                                   |
| Danh sách module  | "dense" (khớp hậu tố, dễ trúng classifier.dense)   | Tên đầy đủ cho 6 module mỗi lớp (72 module)                                                                               |
| Rank khác nhau    | Chỉ r; alpha cố định                               | rank_pattern kèm alpha_pattern, α = 2r                                                                                    |
| Khớp ngân sách    | Tổng rank = n_module × 8 (nhắc "48 module")        | Tổng rank = 96 (12 lớp × 8); ở mức lớp bằng nhau cũng là tham số bằng nhau                                                |
| Largest remainder | Chỉ thêm rank, giả sử tổng floor nhỏ hơn ngân sách | Co giãn trước (tìm hệ số để tổng bằng 96), làm tròn sau, có cận \[1, 32\]                                                 |
| Early stopping    | Patience 5, tối đa 20 epoch                        | Sweep dùng số epoch cố định; early stopping chỉ để cho baseline hiệu năng                                                 |
| Seed              | Mỗi cấu hình seed riêng                            | Split cố định (UIT-VSFC đã chia sẵn); seed xác định khởi tạo head, dropout và thứ tự dữ liệu; cấu hình cùng seed ghép cặp |
| Số lượt           | 337 hay 347                                        | 2.217 lõi (bảng ngân sách trong đề xuất, mục 11)                                                                          |

## 2. Checkpoint, môi trường, phiên bản thư viện

Mô hình chính là XLM-R-base cho cả bốn cell; PhoBERT và RoBERTa chỉ để nhân bản.

| Vai trò                    | Checkpoint                  | Dùng cho |
|----------------------------|-----------------------------|----------|
| Chính (VI và EN)           | FacebookAI/xlm-roberta-base | C1–C7    |
| Nhân bản VI (có điều kiện) | vinai/phobert-base          | C1       |
| Nhân bản EN (có điều kiện) | FacebookAI/roberta-base     | C3       |

- Ghim revision (commit hash) của từng checkpoint khi tải lần đầu và ghi vào file kết quả.

- PhoBERT cần văn bản tiếng Việt đã tách từ (word segmentation) trước khi tokenize. Bản đặc tả cũ mô tả tokenizer là "BPE mức âm tiết", điều này cần kiểm lại theo tài liệu chính thức của mô hình trước khi nhân bản.

- Các mô hình này dùng chung tên module kiểu RoBERTa (encoder.layer.N.attention.self.query, ...). Script tự dò tiền tố (roberta. hay bert.) nên không cần sửa khi đổi checkpoint.

- Phiên bản đã dùng khi kiểm thử trên CPU: torch 2.14.1, transformers 5.18.0, peft 0.21.2, statsmodels 0.15.0. Chạy thật trên GPU phải ghim đúng phiên bản đã dùng cho toàn bộ lưới.

## 3. Cấu hình LoRA theo lớp

Mỗi lớp nhận một rank cho cả 6 module, có alpha bằng 2r, và tên module được liệt kê đầy đủ. Thư viện rankcap.py sinh toàn bộ cấu hình này từ một danh sách 12 rank.

import rankcap  
\# layer 3 bị bóp xuống rank 1, các lớp còn lại giữ 8  
ranks = \[8\] \* 12; ranks\[3\] = 1  
cfg = rankcap.lora_config(model, ranks) \# target_modules + rank_pattern + alpha_pattern  
peft_model = get_peft_model(model, cfg)  
  
\# sweep theo loại module (C7): intermediate.dense rank 32 ở mọi lớp  
cfg = rankcap.lora_config(model, \[8\] \* 12, type_ranks={"intermediate.dense": 32})  
  
\# Dynamic LoRA và bốn đối chứng, cùng ngân sách 96  
ctrl = rankcap.make_controls(sigma, total=96, rng=np.random.default_rng(seed))  
\# ctrl\["uniform" \| "dynamic" \| "random" \| "reversed" \| "permuted"\] -\> mảng 12 rank

Sáu module mỗi lớp: attention.self.query, attention.self.key, attention.self.value, attention.output.dense, intermediate.dense, output.dense. classifier không bao giờ được gắn LoRA; PEFT huấn luyện head phân loại đầy đủ nhờ task_type=SEQ_CLS.

**Phân bổ r_m có khớp ngân sách (allocate_ranks).**

1.  Chuẩn hóa σ_m về \[0, 1\] (min–max).

2.  Tìm hệ số a bằng chia đôi sao cho tổng clip(a · (r_min + (r_max − r_min) · σ_m), r_min, r_max) bằng đúng 96.

3.  Lấy phần nguyên, rồi chia phần dư cho các lớp có phần thập phân lớn nhất (largest remainder), không vượt cận \[1, 32\].

4.  Assert tổng = 96 và mọi rank trong cận.

Ở mức lớp, mỗi đơn vị rank tốn cùng số tham số ở mọi lớp, nên khớp tổng rank cũng là khớp tham số; test test_patterns_and_budget kiểm tra điều này bằng cách đếm tham số LoRA của hai mẫu khác nhau có cùng tổng rank. Nếu sau này phân bổ theo từng module riêng lẻ thì phải khớp bằng count_lora_params, vì một đơn vị rank ở FFN (in + out = 768 + 3072) tốn khoảng 2,5 lần một đơn vị ở attention (768 + 768).

## 4. Siêu tham số huấn luyện

Siêu tham số giữ nguyên cho mọi cấu hình trong một cell, vì thay đổi giữa các cấu hình sẽ lẫn với hiệu ứng của rank.

| Tham số              | Giá trị                                                                                                       | Ghi chú                                                                                                                       |
|----------------------|---------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| Optimizer            | AdamW, betas (0,9; 0,999), eps 1e-8, weight decay 0,01                                                        | Như đặc tả cũ                                                                                                                 |
| Scheduler            | Warmup 10% rồi cosine                                                                                         | Như đặc tả cũ                                                                                                                 |
| Clip gradient        | 1,0                                                                                                           | Như đặc tả cũ                                                                                                                 |
| Precision            | bf16 nếu GPU hỗ trợ                                                                                           | Tự bật trong run_pilot.py                                                                                                     |
| Batch hiệu dụng      | 32                                                                                                            | Giảm batch theo thiết bị thì tăng gradient accumulation để giữ 32 (script hiện dùng batch 32 trực tiếp; chưa có accumulation) |
| LR                   | Grid {1e-4, 2e-4, 3e-4} trên baseline r = 8, chọn Macro-F1 cao nhất trên calibration split pilot, rồi cố định | Chốt ở pilot, dùng cho mọi rank                                                                                               |
| LoRA dropout         | 0,1                                                                                                           | Như đặc tả cũ                                                                                                                 |
| Epoch                | Cố định, dự kiến 6–8, chốt ở pilot bằng đường loss của baseline                                               | Không early stopping trong sweep                                                                                              |
| Max length           | 128                                                                                                           | Tăng lên 256 nếu hơn 5% mẫu bị cắt                                                                                            |
| FFT (mốc downstream) | LR 2e-5, 10 epoch, early stopping patience 3                                                                  | Chỉ dùng cho H7                                                                                                               |

Chọn LR và epoch trên calibration split (val), không chọn trên test. run_pilot.py mặc định đánh giá trên test.csv; khi chạy grid LR, thêm --eval_file calibration.csv. prepare_vsfc.py đổi ba file chia sẵn (train, val, test) thành train.csv, calibration.csv, test.csv và in kết quả cổng G0.

## 5. Quy trình pilot

Pilot chạy 125 lượt trên cell C1 rồi để analyze_pilot.py đưa ra quyết định G1; kết quả ghi vào một CSV có hỗ trợ chạy tiếp nếu bị ngắt.

1.  **Chuẩn bị dữ liệu.** python prepare_vsfc.py --train train.csv --val val.csv --test test.csv --out_dir data/vsfc_sent tạo train.csv, calibration.csv, test.csv (cột text, label; val trở thành calibration). Dùng --text_col text_seg cho PhoBERT.

2.  **Chốt LR và epoch** trên baseline r = 8 với --eval_file calibration.csv (mục 4).

3.  **Chạy sweep cho cả 5 seed trong một lệnh.** Split cố định nên mọi seed dùng chung một thư mục dữ liệu:

python run_pilot.py --data_dir data/vsfc_sent --model FacebookAI/xlm-roberta-base \\  
--cell C1 --sweep --seeds 100 101 102 103 104 --ranks 1 32 --epochs 6 --lr 2e-4 --out results_C1.csv

4.  **Chạy nền và tiếp tục khi bị ngắt.** Chạy lại đúng lệnh cũ, script bỏ qua lượt đã có trong file kết quả. Mỗi seed gồm 25 lượt.

5.  **Phân tích:**

python analyze_pilot.py results_C1.csv --cell C1

Kết quả gồm, theo từng rank: σ²_LR, ρ, p hoán vị của H1, độ tin cậy split-half (Spearman–Brown, có khoảng tin cậy), tỷ lệ lớp vượt 2 sai số chuẩn, profile L_m từ rank 1, quyết định G1, và power hoặc độ tin cậy kỳ vọng nếu giả định ước lượng là sự thật.

**Quy tắc quyết định (tạm thời, khóa sau mô phỏng power).**

- ρ ≥ 0,5 và p \< 0,05: chạy lưới chính C1–C4.

- 0,2 ≤ ρ \< 0,5: chạy C1 và C3, thêm seed, chuyển sang AUC và đơn vị thô.

- ρ \< 0,2: không chạy lưới đầy đủ; chuyển sang nhóm lớp và loại module, báo cáo với tương đương và power.

Lưu ý khi đọc kết quả: pilot chỉ có rank 1 và 32, nên r\*(δ) chưa ước lượng được ở giai đoạn này. Profile chính là L_m từ rank 1. Hai hệ số ρ ở rank 1 và rank 32 có thể khác nhau nhiều, vì rank 32 thường có tín hiệu nhỏ hơn.

## 6. Dữ liệu và giao thức mất cân bằng lớp

Kết quả cổng G0 trên dữ liệu thật: lớp neutral chỉ có 522 mẫu trong train, nên N tối đa cho ρ = 1 là 1.566. Với N = 1.500, cả ba mức ρ = 1, 5, 20 đều khả thi (ở ρ = 20 cần 1.178 / 263 / 59 mẫu cho ba lớp), nhưng N chỉ bằng khoảng 12% train.

**Subsample SST-2 và AG News.** Lấy mẫu phân tầng về cùng quy mô với UIT-VSFC (~16 nghin mẫu), một tập con cố định cho mọi seed.

**Giao thức mất cân bằng.**

1.  Đếm số mẫu từng lớp trong train của mỗi cell.

2.  N tối đa hỗ trợ ρ = 1 là K nhân số mẫu của lớp nhỏ nhất (max_feasible_N). Chọn N chung cho mọi mức ρ.

3.  Kiểm tra ρ ∈ {1, 5, 20} đều khả thi ở N đó (feasible_imbalance): số mẫu cần cho mỗi lớp phải nằm trong số mẫu sẵn có.

4.  Sinh số mẫu mỗi lớp bằng imbalanced_counts(K, N, ρ): n_k tỷ lệ với ρ^(−k/(K−1)), tổng bằng N.

5.  Test set giữ nguyên prior gốc; calibration split theo prior huấn luyện (cần cho Paper B).

6.  Ghi lại trong file kết quả: cell, ρ, N, số mẫu mỗi lớp.

import rankcap  
counts = \[n_neg, n_neu, n_pos\] \# đếm trên train thật  
N = rankcap.max_feasible_N(counts) \# trần cho ρ = 1  
ok = rankcap.feasible_imbalance(counts, N) \# {1: True, 5: ..., 20: ...}  
need = rankcap.imbalanced_counts(3, N, 20) \# số mẫu mỗi lớp ở ρ = 20

Vì N = 1.500 nhỏ, nhiễu seed ở C6 sẽ lớn hơn các cell khác. Lớp neutral là lớp thiểu số (k = K − 1). Cân nhắc làm C6 chính trên AG News (4 lớp cân bằng, đủ mẫu) và dùng UIT-VSFC sentiment làm kiểm tra phụ. Không dùng oversampling để nâng lớp nhỏ, vì nó làm mẫu trùng và thay đổi bản chất của thao túng.

## 7. Kiểm thử đã chạy và checklist triển khai

Chín test đơn vị đã qua trên CPU với mô hình XLM-R thu nhỏ khởi tạo ngẫu nhiên. Ngoài ra run_pilot.py rồi analyze_pilot.py đã chạy xuyên suốt trên dữ liệu UIT-VSFC thật, với tokenizer và mô hình nhỏ dựng cục bộ (sweep, chạy tiếp, --eval_file). Chưa chạy với trọng số tiền huấn luyện vì môi trường kiểm thử bị chặn Hugging Face, nên thời gian mỗi lượt và F1 thật vẫn cần đo trên GPU.

| Test                                   | Kiểm tra điều gì                                                                                                                  |
|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| test_allocate_ranks                    | Tổng rank bằng 96, mọi rank trong \[1, 32\], đơn điệu theo σ, σ hằng cho rank 8 đều                                               |
| test_controls_same_budget              | Năm cấu hình (đều, dynamic, ngẫu nhiên, đảo ngược, hoán vị) cùng tổng 96                                                          |
| test_patterns_and_budget               | 72 module, không trúng classifier, hai mẫu khác nhau có cùng số tham số LoRA, rank đúng từng lớp, scaling α/r bằng 2 ở mọi module |
| test_type_sweep                        | Rank theo loại module áp đúng cho mọi lớp                                                                                         |
| test_imbalance                         | N cố định, tỷ lệ ρ đúng, kiểm tra khả thi                                                                                         |
| test_analysis_detects_signal_and_noise | Dữ liệu giả có tín hiệu cho ρ cao và p nhỏ; dữ liệu thuần nhiễu cho ρ thấp, tỷ lệ p \< 0,05 gần mức kỳ vọng                       |
| test_power_sim_monotone                | Power tăng khi tín hiệu lớn hơn                                                                                                   |
| test_end_to_end_tiny_training          | Vòng huấn luyện và đánh giá chạy được với cấu hình rank theo lớp                                                                  |

**Checklist trước khi chạy.**

- [ ] Ghim revision checkpoint và phiên bản thư viện, ghi vào file kết quả

- [ ] Đếm số mẫu từng lớp và chọn cell cho C6 (cổng G0)

- [ ] Sinh split group-aware cho từng seed; cùng seed dùng chung split cho mọi cấu hình

- [ ] Chạy python test_rankcap.py trên máy GPU

- [ ] Đo thời gian một lượt thật (baseline và một lượt rank 1) và cập nhật bảng ngân sách

- [ ] Chốt LR và số epoch trên calibration split của seed pilot

**Checklist sau pilot.**

- [ ] Chạy analyze_pilot.py, đọc ρ, p, split-half và quyết định G1

- [ ] Mô phỏng power cho H2 đến H5 và chốt ngưỡng định lượng

- [ ] Ghi ngưỡng đã khóa vào kế hoạch thống kê trước khi chạy lưới chính

- [ ] Hạ xuống thăm dò mọi giả thuyết thiếu power (quy tắc G2)

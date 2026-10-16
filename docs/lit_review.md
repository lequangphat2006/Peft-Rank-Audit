# Rà soát tài liệu (lit review)

Cập nhật: 2026-10-16. Trạng thái đọc từng mục: ✅ đã đọc kỹ, ⏳ đã đọc lướt, ⬜ chưa đọc.
Cập nhật ngay trước khi nộp; khi chọn tạp chí thì rà lại theo scope tạp chí.

## 0. Định vị bài này trong một câu

"Các nghiên cứu gần đây cho thấy kết luận về LoRA phụ thuộc mạnh vào learning rate; bài này
kiểm toán có đăng ký trước ảnh hưởng của rank LoRA toàn cục lên Macro-F1 và NLL/ECE, khi
LR được tune công bằng theo từng rank, trên XLM-R-base với tiếng Việt, dưới hai ngân sách
huấn luyện, dùng TOST và 16 seed."

## 1. LoRA và các biến thể rank-adaptive (đối tượng kiểm toán)

| # | Tài liệu | Trạng thái | Vai trò trong bài |
|---|---|---|---|
| 1 | Hu et al. 2022. LoRA: Low-Rank Adaptation of Large Language Models. ICLR. | ✅ | Định nghĩa LoRA, ký hiệu BA, α/r |
| 2 | Zhang et al. 2023. AdaLoRA: Adaptive Budget Allocation for PEFT. ICLR. | ⏳ | Đối tượng: giả định "importance" đo được theo module |
| 3 | Kalajdzievski 2023. A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA (rsLoRA). arXiv:2312.03732. | ⏳ | Sensitivity α ở mức C |
| 4 | IGU-LoRA (2024). Information-Geometric... | ⬜ | Đối tượng (cần xác minh) |
| 5 | BaRA (2024). Bayesian Rank Allocation... | ⬜ | Đối tượng (cần xác minh) |
| 6 | StatLoRA (2024/2025). Statistical... | ⬜ | Đối tượng (cần xác minh) |
| 7 | Unified Study of LoRA Variants. arXiv 2601.22708 (cần xác minh số) | ⬜ | Bối cảnh "kết luận phụ thuộc LR" |
| 8 | Parameter Efficient Instruction Tuning: An Empirical Study. arXiv 2411.16775. | ⏳ | Bối cảnh empirical |
| 9 | Biderman et al. 2024. LoRA Learns Less and Forgets Less. TMLR. | ⏳ | Bối cảnh "LR/rank confound" |
| 10 | Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning (2026). | ⬜ | Bối cảnh gần nhất; phải đọc trước khi nộp |
| 11 | Dettmers et al. 2023. QLoRA. NeurIPS. | ✅ | Bối cảnh |
| 12 | Aghajanyan et al. 2021. Intrinsic Dimensionality Explains the Effectiveness of LM Fine-Tuning. ACL. | ✅ | Nền lý thuyết (capacity thấp) |

## 2. Hiệu chuẩn (calibration)

| # | Tài liệu | Trạng thái | Vai trò |
|---|---|---|---|
| 13 | Guo et al. 2017. On Calibration of Modern Neural Networks. ICML. | ✅ | Temperature scaling, ECE |
| 14 | Ovadia et al. 2019. Can You Trust Your Model's Uncertainty? NeurIPS. | ⏳ | Độ bền hiệu chuẩn dưới dịch chuyển phân phối |
| 15 | Ovadia et al. (same) — dùng như khung OOD | ⏳ | Động cơ cho Giai đoạn 6 (OOD) |
| 16 | (cần bổ sung) một bài gần đây về calibration của PEFT (2024–2025) | ⬜ | Rà khi chọn tạp chí |
| 17 | Lakens 2017. Equivalence Tests: A Practical Primer. SPPS. | ✅ | Nền TOST |
| 18 | Schuirmann 1987. A comparison of the two one-sided tests procedure... | ✅ | Nền TOST gốc |

## 3. Phương pháp luận (pre-registration, kiểm định tương đương)

| # | Tài liệu | Trạng thái | Vai trò |
|---|---|---|---|
| 19 | Nosek et al. 2018. The preregistration revolution. PNAS. | ✅ | Động cơ đăng ký trước |
| 20 | Simmons et al. 2011. False-Positive Psychology. Psych. Science. | ✅ | Động cơ (p-hacking, garden of forking paths) |
| 21 | Gelman & Loken 2014. The Statistical Crisis in Science. | ✅ | Cùng mục đích |
| 22 | Hartung & Knapp 2001. On tests of the overall treatment effect in meta-analysis. | ⏳ | Meta-gộp (mức C) |

## 4. Tiếng Việt và dữ liệu

| # | Tài liệu | Trạng thái | Vai trò |
|---|---|---|---|
| 23 | Nguyen et al. 2018. UIT-VSFC: Vietnamese Students' Feedback Corpus. KSE. | ✅ | R1 |
| 24 | Ho et al. 2020. UIT-VSMEC: Vietnamese Social Media Emotion Corpus. LREC. | ⏳ | B2/B3 |
| 25 | Nguyen et al. 2023. ViSoBERT: A Pre-trained Language Model for Vietnamese Social Media. EMNLP. | ⏳ | B1/B3 |
| 26 | Nguyen & Nguyen 2020. PhoBERT. EMNLP Findings. | ✅ | B1 |
| 27 | Conneau et al. 2020. Unsupervised Cross-lingual Representation Learning at Scale (XLM-R). ACL. | ✅ | Base model |
| 28 | Liu et al. 2019. RoBERTa. arXiv:1907.11692. | ✅ | B5 |

## 5. Task tiếng Anh

| # | Tài liệu | Trạng thái | Vai trò |
|---|---|---|---|
| 29 | Zhang et al. 2015. Character-level Convolutional Networks (AG News). | ✅ | B4 |
| 30 | Socher et al. 2013. Recursive Deep Models (SST-2). EMNLP. | ⏳ | B4 phương án 2 |

## 6. Khoảng trống bài này lấp

1. **Hiệu chuẩn (calibration) dưới rank sweep** — hầu hết bài về rank-adaptive LoRA chỉ báo
   accuracy/F1; NLL và ECE gần như không có. Bài này báo cáo cả trước và sau temperature scaling.
2. **LR tune theo rank** — phần lớn so sánh rank dưới một LR cố định, thiên vị rank thấp.
   Bài này tune LR riêng cho từng rank (5 rank × 4 LR × 3 seed trên calibration).
3. **Ngân sách là biến chính** — 3, 6, 10 epoch và 20%/50% data; H3 (tương tác rank × epoch)
   và hai nhánh LR (theo rank vs cố định) là kiểm định độ bền.
4. **Pre-registration + TOST + 16 seed** — hiếm trong literature PEFT, vốn thường 3–5 seed
   và chỉ dùng NHST.
5. **Encoder nhỏ, tiếng Việt, dữ liệu ít** — bổ sung một điểm dữ liệu cho literature vốn
   tập trung vào LLM tiếng Anh.

## 7. Việc đọc còn mở (trước khi nộp)

- [ ] Xác minh số arXiv của "A Unified Study of LoRA Variants" và bài LR 2026.
- [ ] Tìm 1–2 bài 2024–2025 về calibration của PEFT để so.
- [ ] Đọc IGU-LoRA, BaRA, StatLoRA để định vị chính xác "importance score" bị kiểm toán.
- [ ] Rà theo scope tạp chí khi đã chọn (11-06).
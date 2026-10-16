# Hạn chế



Tổng hợp từ `analysis_plan_v2.0.md`, `pivot_v2.md`, `exploratory_results.md`.

Cập nhật khi phát hiện thêm. Dùng làm nguồn trích dẫn khi viết bài.



---



## 1. Phạm vi



- Một mô hình (XLM-R-base), một ngôn ngữ (tiếng Việt), một task chính (UIT-VSFC sentiment 3 lớp).

- Split cố định — nhiễu do chọn tập huấn luyện không nằm trong phương sai seed.

- Encoder, phân loại văn bản. Không kết luận cho decoder hoặc LLM lớn.



## 2. Dữ liệu



- UIT-VSFC: lớp neutral ~4% train (522 mẫu), ~67 mẫu test.

- Macro-F1 bị chi phối bởi lớp neutral. δ = 0,01 trên Macro-F1 ≈ 1–2 mẫu neutral.

- Near-duplicate (TF-IDF char_wb 3-5 gram, cosine ≥ 0.95):

&#x20; - calibration: 5.19%

&#x20; - test: 3.81%

- Chưa kiểm tra giả định "leak ảnh hưởng mọi rank như nhau" (rank cao có thể ghi nhớ leak nhiều hơn).



## 3. Thiết kế



- LR tune một lần ở 6 epoch, tái dùng cho 3 epoch (không tune lại).

- LR chốt ở lưới {1e-4, 2e-4, 5e-4, 1e-3} — tối ưu có thể ngoài lưới, đặc biệt rank 4 (F1 tăng đơn điệu theo LR).

- α = 2r (α/r = 2 cố định). Độ nhạy α (fixed:16, rsLoRA) ở mức C, chưa chạy.

- Full fine-tuning dùng LR 2e-5 (khác LoRA). LR 2e-4 làm full FT collapse về majority class.



## 4. Kỹ thuật



- **Precision:** fp16 thay vì bf16. T4 = SM 7.5 không có bf16 gốc. `torch.cuda.is_bf16_supported()` trả True nhầm do bug PyTorch. Benchmark: bf16 mô phỏng 21,9s vs fp16 6,7s (3,27×).

- **3 cấu hình collapse ở tune seed 400:** rank 16 + LR 1e-3, rank 32 + LR 5e-4, rank 32 + LR 1e-3. F1 = 0.2207, NLL ≈ 0.837 (khớp entropy lớp train).

- **Reference baselines** dùng seed dải riêng (600–607 cho head, 600–604 cho full FT), không so trực tiếp với R1.



## 5. Nhiễu và power



- SD dư 0,0081 đo từ Stage 1c (calibration, 6 epoch, module-level, bf16).

- Chưa đo trực tiếp trên test + fp16 + LR per rank + rank toàn cục.

- Power 0,89 sau Holm (N = 16, δ = 0,01). Nếu SD thực tế lớn hơn 25% (0,0101), power còn ~0,60.

- 5 lượt full FT đầu dùng LR 2e-4 bị collapse — ghi vào sai lệch, không dùng.



## 6. Đăng ký trước



- Tag git nội bộ (`analysis-plan-v2.0`) + OSF Preregistration (không phải "Registered Report" hay "AsPredicted").

- OSF Preregistration không có peer review trước khi submit, không ràng buộc tạp chí.

- Viết sau khi đã thấy dữ liệu khám phá (module-level). Không phải blind pre-registration.

- Tuy nhiên: seed ranges tách biệt rõ (100–114 exploration, 200–215 confirmation), tag commit hash, không có thay đổi sau tag v2.0.



## 7. Kết luận — điều không được nói



- Không nói "rank không quan trọng" nói chung.

- Không nói "capacity need không tồn tại".

- Không nói "AdaLoRA/IGU-LoRA/BaRA/StatLoRA tối ưu trên nhiễu".

- Không nói "kiểm định đầu tiên".

- Mọi so sánh ghép cặp theo seed, trên cùng tập test. Không khái quát ra ngoài phạm vi đã thử.



## 8. Ghi chú thêm từ peer review nội bộ (2026-10-16)



Các điểm sau đã được ghi nhận và xử lý ở mức thiết kế, nhưng cần nhắc lại trong bài:



- **Căn cứ đổi hướng:** các phân tích khám phá của hướng cũ (Stage 1c) đổi rank một lớp/module, không đo rank toàn cục. Con số gộp F1 −0,0013 không dùng để justify H1 (đó là giả thuyết kiểm chứng được, không phải niềm tin).

- **F1 neutral chiếm ưu thế:** Macro-F1 trên VSFC bị chi phối bởi lớp neutral (~4% train, 67 mẫu test). Báo cáo kèm F1 từng lớp và bootstrap trên test.

- **LR tune trên F1, không trên NLL:** H2 có thể bị confound với LR khác nhau giữa rank. Đã xử lý bằng nhánh LR cố định 2e-4 làm robustness check.

- **Tune LR 1 seed (đến 2026-10-16):** đang chạy thêm seed 401, 402 để chốt LR per rank. Kết luận "LR tối ưu khác nhau giữa rank" chỉ nên nói khi có 3 seed.

- **α = 2r làm rank dính với bước học:** độ nhạy α cần chạy sớm (mức C) để tách ảnh hưởng rank thuần túy khỏi scaling.


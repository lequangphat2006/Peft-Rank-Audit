# EXT-1 — Runbook (hướng dẫn vận hành từng bước)

Ghi chú: thời gian ước lượng dùng ~260 giây/lượt 6 epoch (đo thực tế R1: 257–263 s; B1 PhoBERT ~245 s) trên T4. Là cận trên vì nhiều lượt được dùng lại nhờ resume.

## 0. Chuẩn bị (một lần)
1. Lấy lại mã và dữ liệu: `peft-rank-audit` (zip này), thư mục dữ liệu `vsfc_sent` (XLM-R) và bản PhoBERT đã tách từ (cột `text` = `text_seg`; dùng đúng thư mục đã dùng cho B1 cũ).
2. Cài đúng phiên bản: `peft==0.21.2`, `transformers==5.16.1` (như R1). Không trộn fp16/fp32 trong một so sánh.
3. Đặt biến: `DATA_P`, `DATA_X`, `NGPU` (xem đầu `scripts/run_ext_phobert.sh`).
4. Chạy `python -m pytest code/test_ext_analysis.py -q` (14 kiểm thử, không cần GPU).
5. Chạy `bash scripts/run_ext_phobert.sh preflight_pre`.
6. **Cập nhật đăng ký OSF** bằng `docs/ext_osf_amendment.md`, gắn tag git `analysis-plan-v2.1` **trước khi chạy lượt đầu tiên**.

## 1. Mức P1 (bắt buộc) — tune LR rồi chạy chính
| Bước | Lệnh | Số lượt | Giờ GPU (cận trên) |
|---|---|---|---|
| 1 | `bash scripts/run_ext_phobert.sh tune_base` | 60 | 4,3 |
| 2 | `bash scripts/run_ext_phobert.sh lr_rule_base` → in rank cần mở rộng | — | — |
| 3 | `EXT_LOW_RANKS="…" EXT_HIGH_RANKS="…" bash scripts/run_ext_phobert.sh tune_ext` | ≤ 15 (thường 6–9) | ≤ 1,1 |
| 4 | `bash scripts/run_ext_phobert.sh lr_rule_final` → `phobert_lr_map_v2.json` đủ 5 rank | — | — |
| 5 | `bash scripts/run_ext_phobert.sh preflight_main` (phải báo 0 lỗi, dry_run đúng 80 lượt) | — | — |
| 6 | `bash scripts/run_ext_phobert.sh final` | 80 | 5,8 |
| 7 | `bash scripts/run_ext_phobert.sh merge_final` (phải báo `complete=True`) | — | — |
Tổng P1 ≤ 155 lượt ≈ 11,2 giờ GPU (≈ 5,6 giờ thực với 2 GPU).

## 2. Mức P2 (rất nên) — độ bền
| Nhóm | Lệnh | Số lượt tối đa | Giờ GPU |
|---|---|---|---|
| LR cố định 2e-4 | `… robust_fixed` | PhoBERT 40 + XLM-R 16 | 4,0 |
| LR tốt nhì | `… robust_second` | PhoBERT 40 + XLM-R 40 | 5,8 |
| Độ nhạy α | `… alpha` | 24 + 24 | 3,5 |
Tổng P2 ≤ 184 lượt ≈ 13,3 giờ GPU. **P1 + P2 ≈ 24,5 giờ GPU** (≈ 12 giờ thực với 2 GPU).

## 3. Mức P3 (tùy chọn)
`… three_epoch`: PhoBERT 3 epoch, 80 lượt, ≈ 2,9 giờ (3 epoch ≈ 132 s/lượt). Chỉ làm nếu cô muốn so H3 cho PhoBERT.

## 4. Phân tích
`bash scripts/run_ext_phobert.sh analyze` gọi lần lượt: `ext_analyze_phobert.py` (mô tả + độ bền), `ext_confirmatory.py` (E1–E3, Holm, bootstrap hai tầng, quy tắc kết luận), `ext_calibration.py`, `ext_deltaw.py`, `ext_alpha.py`, `ext_make_figures.py`. Kết quả vào `results/ext/phobert/` và `figures/fig13–fig15`.

## 5. Điểm dễ sai
- **LR map thiếu rank → rank bị bỏ qua im lặng.** Luôn chạy `preflight_main` trước `final`.
- **Seed tune ≠ seed chạy chính.** Preflight kiểm tra.
- **Resume theo `run_id`**: chạy lại một giai đoạn là an toàn; seed được chia GPU theo `seed % NGPU` nên không chạy trùng giữa các giai đoạn. Nếu đổi `NGPU` giữa chừng, gộp bằng `ext_merge.py` (khử trùng run_id) và chấp nhận có thể chạy dư vài lượt.
- **Logits và ΔW của B1 cũ không có trong repo** → bootstrap ghép cặp và hiệu chuẩn cho PhoBERT chỉ làm được với B1X (đã chạy lại, có đủ logits).
- **Sập (F1 < 0,25)**: giữ nguyên, không chạy lại để "chữa".
- **Không thêm seed sau khi xem kết quả.**

## 6. Tiến độ gợi ý (hôm nay 26/10/2026, hạn cuối năm)
| Tuần | Việc |
|---|---|
| 26/10–1/11 | OSF + tag v2.1; test; preflight; bắt đầu `tune_base` |
| 2/11–8/11 | `lr_rule_base`, `tune_ext`, `lr_rule_final`; bắt đầu `final` |
| 9/11–22/11 | hoàn tất `final`, P2 (`robust_*`, `alpha`) |
| 23/11–6/12 | `analyze`, hình, bảng; rà soát |
| 7/12–20/12 | viết bài, gửi cô góp ý |
| 21/12–31/12 | dự phòng; P3 nếu cần; chỉnh sửa |
Kiểm tra hạn mức GPU Kaggle theo tuần: tuần 2 và 3 dùng nhiều nhất.

## 7. Danh sách kiểm tra trước khi nộp
- [ ] Tag `analysis-plan-v2.1` có ngày trước lượt chạy đầu tiên; OSF đã cập nhật.
- [ ] Mọi lượt cùng commit và phiên bản thư viện; cột `git` có đúng một giá trị.
- [ ] `merge_final` báo `complete=True` (80 lượt); manifest đã commit.
- [ ] Báo cáo cả kết quả 8 seed và 16 seed; cả lượt sập.
- [ ] Bảng đồng nhất thiết kế (A) và yếu tố gây nhiễu (B) có trong bài.
- [ ] H3 ghi rõ chỉ kiểm cho XLM-R (nếu không chạy P3).
- [ ] Sửa các câu quá lời trong CHANGELOG/bản thảo ("chứng minh", "H3 bền vững").

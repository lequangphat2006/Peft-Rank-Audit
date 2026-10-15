# Lộ trình v2 — Rank LoRA, khả năng phân biệt và hiệu chuẩn

> Hướng đề tài này thay thế hướng "capacity need" ban đầu; lý do và so sánh xem `docs/pivot_v2.md`. Số seed, δ và chỉ số theo `docs/analysis_plan.md` (nguồn chuẩn khi có khác biệt).

*Cập nhật: 2026-10-15. Thay cho các lộ trình trước. Mọi con số thời gian là ước tính từ pilot (~138 giây/epoch/lượt trên một T4: 3 epoch ≈ 6,9 phút, 6 epoch ≈ 13,8 phút) và phải đo lại ở lượt đầu tiên.*

## 1. Câu hỏi nghiên cứu (đã đổi)

Từ "capacity need có đo được không" sang: **rank LoRA có ảnh hưởng gì đến Macro-F1 và NLL, và hiệu ứng đó có phụ thuộc ngân sách huấn luyện không?** Lý do: dữ liệu khám phá hiện có cho thấy F1 không đổi theo rank (−0,0013 [−0,0051; +0,0025]) trong khi NLL tệ nhẹ (+0,0071 [+0,0035; +0,0107]), và không phát hiện khác biệt giữa module. Tất cả còn là khám phá cho đến khi xác nhận trên seed mới.

## 2. Việc cần quyết trước khi chạy (Giai đoạn 0)

1. **Loss.** Code hiện hành dùng cross-entropy không trọng số (`model(..., labels=y).loss`). README và `docs/decisions.md` ghi "class-weighted CE". Hai điều này mâu thuẫn. Đề xuất giữ không trọng số (khớp với mọi kết quả đã có) và sửa docs. Cờ `--class_weight balanced` có sẵn nếu muốn thử riêng như một thí nghiệm.
2. **Precision trên T4.** `auto` chọn bf16 khi `torch.cuda.is_bf16_supported()` trả True. T4 không có bf16 gốc nên có thể đang chạy bf16 giả lập (chậm). Chạy một lệnh kiểm tra trên Kaggle: `python -c "import torch;print(torch.cuda.get_device_capability(0), torch.cuda.is_bf16_supported())"`. Nếu cần đổi sang `--precision fp16`, chỉ đổi cho **toàn bộ** Giai đoạn 1 (không trộn), và coi các kết quả cũ chỉ để tham khảo.
3. **δ và chỉ số chính.** Xem `docs/analysis_plan.md` mục 4. Gửi TS. Huyên xem trước khi khóa.
4. **Nhãn topic.** Kiểm tra file thô UIT-VSFC có cột topic và phân bố lớp (cần để chạy R2). `run_pilot.py` v2 đã hỗ trợ K ≠ 3.

## 3. Giai đoạn 0 — Chuẩn bị (2–3 ngày, 0 GPU)

| Bước | Việc | Kết quả |
|---|---|---|
| 0.1 | ĐÃ XONG: `run_pilot.py` v2, `test_run_pilot.py`, `analyze_global.py`, `power_global_sweep.py` đã có trong `code/`; `analysis_plan.md`, `roadmap_v2.md` đã có trong `docs/`. Bản v1 không đổi tên thành file riêng mà truy vết bằng lịch sử git (commit trước `Add Stage 1c v2/v3/v4 analysis + global sweep + docs`) | Repo có v1 để truy vết |
| 0.2 | `python -m pytest code/test_run_pilot.py code/test_rankcap.py -v` trên máy local (CPU, ~1–2 phút) | 15 test đạt |
| 0.3 | Quyết định mục 2 (loss, precision, δ). Loss và δ đã chốt và docs đã sửa (2026-10-15); precision và nhãn topic còn mở (`pivot_v2.md` mục 8) | Docs khớp code |
| 0.4 | TS. Huyên xem `analysis_plan.md`; commit và `git tag analysis-plan-v1` | Kế hoạch khóa, có ngày |
| 0.5 | ĐÃ XONG (2026-10-15): `python code/power_global_sweep.py results/pilot/stage1c_merged.csv --delta_f1 0.005 0.0075 0.01 --nll_effect 0.007`. Kết quả: δ = 0,005 không khả thi (power tối đa 0,19 ở N = 16); δ = 0,01 đạt 0,95 ở N = 16; N = 16 cho power NLL 0,87 (N = 12: 0,772) | N = 16 seed, δ = 0,01 |
| 0.6 | ĐÃ XONG (2026-10-15, PASS): 2 lượt rank 1 và 32, 3 epoch, cell `SMOKE`; ~398 và ~468 giây/lượt; có `logits/*.npz`, `epochwise/*.json`, `.meta.jsonl` | Thời gian thật khớp ước tính ~400 giây |

## 4. Giai đoạn 1 — Global sweep xác nhận (≈ 22 GPU-giờ cho 1a; ≈ 25 kể cả 1b)

**1a. Thiết kế:** rank {1, 4, 16, 32} × epoch {3, 6} × seed 200–215 (16 seed), cell `R1`. 128 lượt, ≈ 22,1 GPU-giờ, ≈ 11 giờ thực trên 2×T4. Chạy hai đợt bằng nhau: seed 200–207 (64 lượt), rồi seed 208–215 (64 lượt; đổi danh sách `--seeds` trong lệnh dưới), **không phân tích H1–H3 giữa hai đợt**. Sau đợt 1 chỉ được chạy `power_global_sweep.py` để xem SD dư; nếu SD dư > 0,009 thì mở rộng lên 20 seed (216–219) theo `analysis_plan.md` mục 6.

```bash
# Kaggle, mỗi GPU một tiến trình (chạy lại đúng lệnh này nếu bị ngắt; run_id giúp bỏ qua lượt đã xong)
cd /kaggle/working/code
CUDA_VISIBLE_DEVICES=0 python run_pilot.py --data_dir /kaggle/working/data/vsfc_sent --model FacebookAI/xlm-roberta-base \
  --cell R1 --sweep_global --ranks 1 4 16 32 --epochs_list 3 6 --seeds 200 201 202 203 204 205 206 207 \
  --shard 0/2 --out /kaggle/working/results/global_gpu0.csv &
CUDA_VISIBLE_DEVICES=1 python run_pilot.py --data_dir /kaggle/working/data/vsfc_sent --model FacebookAI/xlm-roberta-base \
  --cell R1 --sweep_global --ranks 1 4 16 32 --epochs_list 3 6 --seeds 200 201 202 203 204 205 206 207 \
  --shard 1/2 --out /kaggle/working/results/global_gpu1.csv &
wait
```

Gộp: `pd.concat([pd.read_csv(p) for p in (gpu0, gpu1)])` rồi lưu `global_merged.csv` (v2 có cột `run_id`; không dùng `merge_results.py` cũ nếu nó giả định cột v1).

Phiên Kaggle tối đa ~12 giờ và có hạn mức GPU hàng tuần: hãy kiểm tra hạn mức còn lại trước khi chạy và lưu `/kaggle/working/results` ra Dataset sau mỗi đợt.

**1b. Kiểm tra LR (≈ 2,3 GPU-giờ):** thêm `--lr_list 2e-4 5e-4 --ranks 1 32 --epochs_list 6 --seeds 200 201 202 203 204 --cell R1`. Các lượt LR 2e-4 đã có sẽ được bỏ qua tự động (cùng `run_id`). Mục đích: kiểm tra rank nhỏ có bị thiệt vì LR chốt ở rank 8 hay không. Thăm dò, không thuộc họ xác nhận.

**Phân tích sau 16 seed:**
```bash
python code/analyze_global.py results/global/global_merged.csv --primary_epochs 6 --delta_f1 0.01 \
  --logits_dir results/global/logits --out results/analysis/global_v1.json
```

## 5. Giai đoạn 2 — Nhân bản trên task thứ hai (≈ 7,4 GPU-giờ)

UIT-VSFC **topic** (cùng văn bản và mô hình, khác nhãn), cell `R2`, rank {1, 4, 16, 32}, 6 epoch, seed 300–307 (32 lượt). Dùng cùng `analyze_global.py` (chỉ có một ngân sách nên H3 bỏ). Đây là nhân bản có kế hoạch, không điều chỉnh gì sau khi nhìn kết quả R1. Nếu file thô không có nhãn topic, dùng AG News và ghi rõ rằng ngôn ngữ và task bị lẫn.

## 6. Giai đoạn 3 — Phân tích hiệu chuẩn (0 GPU)

Temperature scaling đã có trong `analyze_global.py` (phần thăm dò). Đọc: nếu hiệu ứng rank lên NLL biến mất sau chỉnh nhiệt độ thì đó là overconfidence, không phải dung lượng. Nội dung này liên quan trực tiếp đến Paper B (calibration của PEFT), nên cân nhắc đưa vào Paper B nếu quá dài cho Paper A.

## 7. Cổng quyết định sau Giai đoạn 1

| Kết quả | Hành động |
|---|---|
| H1 đạt, H2 đạt | Bài chính: "rank không ảnh hưởng phân biệt, hơi hại hiệu chuẩn trong thiết lập này". Chạy R2 |
| H1 không đạt, F1 tăng theo rank | Rank toàn cục có tác dụng → mới đáng làm lại phân bổ theo module (dùng Stage 1c và `--sweep_types`) |
| H3 đạt | Bài chính: "độ nhạy rank phụ thuộc ngân sách". Thêm ngân sách 10 epoch cho 2 rank (1 và 32) để vẽ đường cong |
| H1 không đạt và CI rộng | Báo cáo "chưa đủ bằng chứng"; chỉ thêm seed nếu đã khai báo ở mục 6 của analysis_plan |
| H2 không đạt | Bỏ khẳng định về NLL, giữ F1 |

## 8. Việc không làm

- Không chạy 12 epoch; không dùng lại seed 100–114 cho xác nhận.
- Không đổi chỉ số chính, δ hay họ kiểm định sau khi thấy dữ liệu mới; mọi thay đổi vào mục 9 của `analysis_plan.md`.
- Không mở rộng sang module-level trước khi biết rank toàn cục có tác dụng.
- Không viết "capacity need không tồn tại", "AdaLoRA tối ưu trên nhiễu", "kiểm định đầu tiên".

## 9. Bài viết

Phạm vi phải ghi rõ: một mô hình (XLM-R-base), tiếng Việt, hai task, 3–6 epoch, LR cố định. Cấu trúc gợi ý: (1) động cơ và câu hỏi, (2) thiết kế và kế hoạch đăng ký trước, (3) phát hiện khám phá (nêu rõ là khám phá), (4) kết quả xác nhận, (5) hiệu chuẩn, (6) hạn chế (một mô hình, split cố định, LR chốt ở rank 8, class imbalance ~4% lớp neutral). Chưa cam kết hạng tạp chí; xác định nơi nộp sau khi có kết quả xác nhận. Rà tài liệu (IGU-LoRA, BaRA, StatLoRA, RSLoRA) cập nhật ngay trước khi nộp.

## 10. Chi phí và tiến độ

| Hạng mục | Lượt | GPU-giờ | Thời gian thực (2×T4) |
|---|---|---|---|
| 1a Global sweep (16 seed × 4 rank × 2 ngân sách) | 128 | ~22,1 | ~11 giờ |
| 1b Kiểm tra LR (chỉ lượt LR 5e-4 mới) | 10 | ~2,3 | ~1,2 giờ |
| 2 Nhân bản VSFC topic | 32 | ~7,4 | ~3,7 giờ |
| **Tổng** | 170 | **~32** | **~16 giờ** |

Nếu SD dư sau đợt 1 vượt 0,009 và phải mở rộng lên 20 seed (216–219): thêm 32 lượt (~5,5 GPU-giờ). Thời gian ước tính từ ~138 giây/epoch/lượt; smoke test cho ~398–468 giây/lượt 3 epoch.

| Tuần | Việc |
|---|---|
| 1 | Giai đoạn 0: sửa docs, test, TS. Huyên duyệt, khóa kế hoạch, smoke test Kaggle |
| 2 | Giai đoạn 1a (2 đợt) + 1b |
| 3 | Phân tích H1–H3; Giai đoạn 2 (R2); Giai đoạn 3 |
| 4–6 | Viết bản thảo; bảng, hình; rà tài liệu |

## 11. Checklist trước mỗi đợt chạy

- [ ] `git status` sạch và commit hash ghi trong CSV (`git` không có `+dirty`).
- [ ] Cùng `--precision`, `--lr`, `--class_weight` cho mọi lượt của một so sánh.
- [ ] Mỗi tiến trình GPU có `--shard i/n` riêng và `--out` riêng.
- [ ] Hạn mức GPU Kaggle còn đủ; kết quả được lưu ra Dataset sau mỗi đợt.
- [ ] Số lượt trong CSV = số lượt kỳ vọng (128 cho 1a); không có `run_id` trùng.

## Phụ lục — mục CHANGELOG để dán

```
## 2026-10-15 — Cycle 2a (Chuẩn bị Giai đoạn 1: global rank sweep)
- run_pilot.py v2: --sweep_global, --shard, --epochs_list/--lr_list, đánh giá calibration + test,
  lưu logit (.npz) và epochwise (.json), run_id làm khóa resume, metadata (git, precision, versions).
- Phát hiện: docs ghi class-weighted CE nhưng code dùng CE không trọng số (đã thống nhất: giữ CE không trọng số, sửa docs).
- Thêm: analyze_global.py, power_global_sweep.py, docs/analysis_plan.md, docs/roadmap_v2.md.
- Kết quả khám phá (chưa xác nhận): F1 gộp rank 1->32 = -0.0013 [-0.0051, +0.0025]; NLL = +0.0071 [+0.0035, +0.0107].
```

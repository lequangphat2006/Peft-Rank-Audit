# Đổi hướng đề tài (2026-10-15): từ "capacity need" sang "rank LoRA toàn cục"

Tài liệu này giải thích đề tài đã đổi như thế nào so với bản đề xuất 4 và đặc tả triển khai bản 2 (ngày 2026-10-03), vì sao đổi, cái gì được giữ lại, cái gì bị bỏ, và những việc còn mở. Mọi con số lấy từ `CHANGELOG.md` và `results/analysis/`.

Trạng thái: **đề xuất nội bộ, chưa được TS. Đinh Thị Hồng Huyên duyệt.** Tên đề tài mới ở mục 6 là đề xuất, không phải tên đã đăng ký.

---

## 1. Tóm tắt trong năm dòng

1. Đề tài ban đầu hỏi: *capacity need của một lớp LoRA có đo được vượt nhiễu seed, ổn định và chuyển giao được không?*
2. Pilot (125 lượt) cho cổng G1 = NO-GO ở mức từng lớp; các phân tích thô hơn (nhóm lớp, loại module) cũng không thấy tín hiệu với 5 seed.
3. Một tín hiệu ở `intermediate.dense` (ρ = 0,88, 2 seed) biến mất khi tăng số epoch và khi tăng seed (ρ = 0,008 ở 5 seed).
4. Vì vậy câu hỏi đổi thành: *rank LoRA toàn cục ảnh hưởng gì đến Macro-F1 và NLL, và hiệu ứng đó có phụ thuộc ngân sách huấn luyện không?* Đây vẫn là một kiểm toán (không đề xuất thuật toán), nhưng hẹp hơn và có đăng ký trước.
5. Đây **không phải** "vẫn theo kế hoạch": kế hoạch cũ có nhánh dự phòng cho ρ < 0,2, nhưng nhánh đó (nhóm lớp, loại module) cũng âm tính, nên đề tài đã đi ra ngoài kế hoạch.

---

## 2. Diễn biến

| Ngày | Sự kiện | Kết quả | Nguồn |
|---|---|---|---|
| 10-03 | Bản đề xuất 4 và đặc tả bản 2 | Pilot làm cổng quyết định trước ~420 giờ GPU | file docx |
| 10-04 | Pilot C1, 125 lượt (rank 1 và 32, 3 epoch, seed 100–104) | G1 **NO-GO** ở mức lớp: σ²_LR = 0, ρ = 0, power = 4,6% (hiệu ứng ~0,006 so với nhiễu ~0,013) | CHANGELOG, `G1_report.txt` |
| 10-04 | Đơn vị thô: 3 nhóm lớp; 6 loại module | ICC nhóm lớp 0,03; ICC module tối đa 0,26 (`intermediate.dense`) | `G1_coarse.txt`, `G1_type*.txt` |
| 10-04 | Power analysis | 5 seed chỉ phát hiện hiệu ứng ≥ 0,02 | `power_analysis.txt` |
| 10-05 | Stage 1a: dose-response, 2 seed | `intermediate.dense` đơn điệu (ρ = 0,88, p = 0,004) | `G1_stage1a.json` |
| 10-05 | Stage 1b: 3 so với 6 epoch | Khoảng cách rank 1 so với 32 giảm ~6 lần (+0,020 xuống +0,0035) | CHANGELOG |
| 10-14/15 | Stage 1c: 6 module × 4 rank × 5 seed mới (105–109), 6 epoch | Không module nào có ý nghĩa (Holm = 1,0); ρ của `intermediate.dense` = 0,008 | `G1_stage1c.json`, `stage1c_v2..v4.json` |
| 10-15 | Smoke test v2 trên Kaggle T4 | Pass; ~400 giây/lượt 3 epoch; tỷ lệ tham số rank 32 so với rank 1 = 32,0 | CHANGELOG |
| 10-15 | `roadmap_v2.md`, `analysis_plan.md`, power analysis cho global sweep | Hướng mới, δ = 0,01, N = 16 seed | `docs/` |

Số liệu khám phá gộp (6 epoch, seed 105–109, mọi module): hệ số F1 theo log2(rank) nhân 5 cho tổng 1→32 là **−0,0013 [−0,0051; +0,0025]**; NLL **+0,0071 [+0,0035; +0,0107]**. Đây là khám phá, không phải xác nhận.

---

## 3. So sánh với đề tài ban đầu

| Hạng mục | Bản 4 / đặc tả bản 2 (10-03) | Hiện tại (10-15) |
|---|---|---|
| Câu hỏi trung tâm | Capacity need của lớp có đo được, ổn định, chuyển giao được không? | Rank toàn cục ảnh hưởng gì đến Macro-F1 và NLL; có phụ thuộc ngân sách huấn luyện không? |
| Đơn vị | 12 lớp, rồi loại module | Toàn cục (cùng rank cho 12 lớp × 6 module). Module chỉ còn là phân tích thăm dò E6 |
| Giả thuyết | H1–H7 (đo được, ổn định, cộng tính, chuyển giao, proxy, mất cân bằng, phân bổ oracle) | H1–H3 mới (tương đương F1, NLL tăng theo rank, tương tác rank × epoch). Xem bảng ánh xạ ở mục 4 |
| Mô hình / ngôn ngữ | XLM-R-base trên VI và EN; PhoBERT, RoBERTa để nhân bản | XLM-R-base, tiếng Việt |
| Task | 4 cell (C1–C4), sau đó C5–C7 | R1: UIT-VSFC sentiment (chính); R2: nhân bản trên task thứ hai |
| Mất cân bằng lớp | Thao túng ρ ∈ {1, 5, 20} | Không thao túng; chỉ nêu là hạn chế (lớp neutral ~4%) và báo cáo F1 từng lớp (E3) |
| Proxy (Fisher, uncertainty) | RQ2 | Bỏ. Có thể quay lại nếu rank toàn cục có tác dụng và module-level đáng làm |
| Nhiễu | Phân rã phương sai lớp × rank so với seed | Hồi quy `metric ~ C(seed) + log2(rank)`, TOST, F-test tương tác |
| Chỉ số | Macro-F1, ΔF1 | Macro-F1 (chính), NLL (chính), ECE sau temperature scaling (phụ) |
| Ngân sách huấn luyện | Cố định 6–8 epoch | Hai ngân sách: 3 và 6 epoch (là biến của thí nghiệm) |
| Seed | 5 (pilot) | 16 seed xác nhận (200–215), không dùng lại 100–114 |
| Tính toán | ~420 giờ lõi | ~32 GPU-giờ (xem `roadmap_v2.md` mục 10) |
| Bài báo | Paper A (RQ1–RQ3) và Paper B (calibration, Pareto) | Một bài kiểm toán; phần hiệu chuẩn có thể tách sang Paper B |

---

## 4. Ánh xạ giả thuyết

Bản 4 dùng nhãn H1–H7. Họ giả thuyết mới cũng dùng H1–H3 nhưng **nghĩa khác**. Để tránh nhầm, tài liệu hiện hành chỉ dùng H1–H3 theo nghĩa mới; khi nhắc bản 4, viết "H1 (bản 4)" v.v.

| Bản 4 | Nội dung | Số phận |
|---|---|---|
| H1 (bản 4) | Phương sai tương tác lớp × rank > 0 | Đã kiểm tra ở pilot: không thấy (power 4,6%). Giữ làm kết quả khám phá, không còn là giả thuyết xác nhận |
| H2 (bản 4) | Profile ổn định qua seed | Split-half 0,14 (rank 1) và 0,49 (rank 32), CI rất rộng. Không còn kiểm |
| H3 (bản 4) | Tác động các lớp cộng tính | Không chạy |
| H4 (bản 4) | Profile chuyển giao qua task/ngôn ngữ | Không chạy (không có profile ổn định để chuyển giao) |
| H5, H6 (bản 4) | Proxy dự đoán profile; mất cân bằng làm giảm proxy | Không chạy |
| H7 (bản 4) | Phân bổ oracle/proxy vượt baseline | Không chạy; chỉ xét lại nếu H1 mới không đạt và F1 tăng theo rank |

| Mới | Nội dung | Kiểm định |
|---|---|---|
| H1 | Hiệu ứng tổng rank 1→32 lên Macro-F1 tương đương 0 ở 6 epoch | TOST, δ = 0,01, Holm m = 3 |
| H2 | NLL tăng theo rank ở 6 epoch | Hệ số > 0, một phía, Holm m = 3 |
| H3 | Độ dốc rank của F1 khác nhau giữa 3 và 6 epoch | F-test tương tác, Holm m = 3 |

---

## 5. Cái được giữ lại và cái bị bỏ

**Giữ lại (dùng cho hướng mới):**
- `code/rankcap.py` (cấu hình LoRA theo lớp, `rank_pattern`, `alpha_pattern`, khớp ngân sách), `run_pilot.py` v2 (có `--sweep_global`, `--shard`, lưu logit, chạy tiếp), `prepare_vsfc.py`.
- Dữ liệu UIT-VSFC sentiment (12.780 / 1.598 / 1.599), LR 2e-4 (chốt bằng grid ở rank 8), hạ tầng Kaggle 2×T4.
- Toàn bộ kết quả pilot, Stage 1a–1c: giữ nguyên làm dữ liệu khám phá, ghi trong sổ phân tích (`analysis_plan.md` mục 8).

**Bị bỏ hoặc hoãn:**
- Ma trận C1–C7 (đa task, đa ngôn ngữ, mất cân bằng) và nhân bản PhoBERT/RoBERTa.
- Proxy (Fisher, uncertainty, effective rank, intrinsic dimension) và lý thuyết điều kiện hoạt động (đóng góp 2 và 3 trong README cũ).
- Stage 1d (can thiệp theo nhóm lớp, 3 nhóm × 2 rank × 10 seed): có trong `decisions.md` nhưng chưa chạy, bị thay thế bởi global sweep.
- Cổng G1 vẫn tồn tại trong lịch sử, nhưng không còn là cổng của kế hoạch hiện hành.

---

### Chi tiết của hướng cũ được giữ ở đâu

Để không mất những gì quan trọng khi đổi hướng:

| Nội dung | Vị trí |
|---|---|
| Toàn văn đề xuất bản 4 (định nghĩa formal g_m(r), r*_m(δ), L_m, mô hình nhiễu và ρ, bảy giả thuyết và kiểm định, thiết kế C1–C7, ngân sách ~420 giờ, kịch bản diễn giải) | `docs/archive/proposal_v4_capacity_need.md` |
| Toàn văn đặc tả bản 2 (so sánh với đặc tả cũ, checkpoint, cấu hình LoRA theo lớp, `allocate_ranks`, siêu tham số, quy trình pilot và quy tắc G1, giao thức mất cân bằng, checklist) | `docs/archive/spec_v2_pilot_and_grid.md` |
| README trước khi đổi (ba đóng góp, bảng RQ gốc, lệnh pilot, bảng quyết định) | `docs/archive/README_pre_pivot.md` |
| Số liệu và chú thích của pilot và Stage 1a–1c, kèm cách đọc đúng | `docs/exploratory_results.md` |
| Diễn biến theo thời điểm | `CHANGELOG.md` (không sửa các mục cũ) |
| Báo cáo gốc, JSON, hình | `results/analysis/` |

Ba đóng góp của README cũ (đã lưu trữ): (1) định nghĩa formal *capacity need* và đo bằng rank sweep; (2) kiểm chứng proxy: uncertainty so với effective rank so với intrinsic dimension; (3) lý thuyết điều kiện hoạt động: khi nào proxy hợp lý, khi nào ngân sách tham số quan trọng hơn cách phân bổ.

Các hàm trong `code/rankcap.py` phục vụ hướng cũ vẫn còn và còn được test: `allocate_ranks` (co giãn trước, làm tròn sau, cận [1, 32], tổng rank = 96), `make_controls` (uniform, dynamic, random, reversed, permuted), `max_feasible_N`, `feasible_imbalance`, `imbalanced_counts` (giao thức mất cân bằng ρ ∈ {1, 5, 20}), `count_lora_params`. Hướng mới chỉ dùng `lora_config` với rank đều.

Hai điều từ bản 4 vẫn nên giữ ngay cả trong hướng mới: (a) trần độ tin cậy, tức một proxy không thể dự đoán một profile tốt hơn chính độ tin cậy của profile đó; (b) quy tắc "một giả thuyết mà mô phỏng cho thấy không đủ power sẽ bị hạ xuống thăm dò trước khi chạy". Điều (b) tương ứng với quy tắc dừng và bảng diễn giải trong `analysis_plan.md` (mục 6 và 10). Điều (a) chưa có chỗ trong kế hoạch mới (không còn profile để dự đoán); chỉ cần nhớ nếu quay lại câu hỏi phân bổ theo module.

## 6. Hệ quả cho tên đề tài (đề xuất, cần TS. Huyên duyệt)

Tên cũ ("kiểm toán đa tác vụ, đa ngôn ngữ, nhận biết mất cân bằng lớp về phân bổ rank") **không còn khớp** với nội dung. Đề xuất:

- Tiếng Việt: *Rank LoRA toàn cục có cải thiện khả năng phân biệt và hiệu chuẩn? Một kiểm toán đăng ký trước trên XLM-R với UIT-VSFC.*
- Tiếng Anh: *Does Global LoRA Rank Matter? A Pre-Registered Audit of Discrimination and Calibration Under Fixed Training Budgets.*

Nếu tên cũ đã được đăng ký hay phê duyệt ở đâu đó, cần báo cô về việc thu hẹp phạm vi.

---

## 7. Những điều được và không được nói

Theo `analysis_plan.md` mục 10 và `roadmap_v2.md` mục 8:

- **Được nói:** "Trong thiết lập này (XLM-R-base, UIT-VSFC, LR cố định, 3 và 6 epoch), với 5 seed và các đơn vị đã thử, không phát hiện capacity need đo được vượt nhiễu seed; power ở mức lớp là 4,6%."
- **Không được nói:** "Capacity need không tồn tại"; "AdaLoRA/IGU-LoRA tối ưu trên nhiễu"; "kiểm định đầu tiên"; "rank không quan trọng" nói chung. Power của kiểm tra tương tác module × rank ở Stage 1c chưa biết.

---

## 8. Việc còn mở trước khi chạy Giai đoạn 1

| # | Việc | Ai quyết | Trạng thái |
|---|---|---|---|
| 1 | Tag `analysis-plan-v2.0` chính thức | Bạn (toàn quyền từ TS. Huyên) | 2026-10-17 |
| 2 | Precision trên T4: chốt **fp16** | Bạn | 2026-10-16 (benchmark 3,27×) |
| 3 | VSMEC: dùng github UIT hoặc fallback | Bạn | Chưa (cần trước 11-04) |
| 4 | Chỉ số đánh giá chính: test | Bạn | Đã chốt |
| 5 | Loss: CE không trọng số | Bạn | Đã chốt |

---

## 9. Nhật ký chỉnh sửa tài liệu cho nhất quán (2026-10-15)

| Tài liệu | Chỗ sửa | Lý do |
|---|---|---|
| `README.md` | Viết lại giới thiệu, đóng góp, trạng thái, bảng RQ, cấu trúc repo, loss, lệnh chạy, bảng quyết định, giới hạn | Phản ánh hướng mới; README còn mô tả đề tài ban đầu |
| `README.md`, `docs/decisions.md` | "Class-weighted CE" đổi thành CE không trọng số | Code dùng `--class_weight none` mặc định; mọi kết quả đã có không dùng trọng số |
| `README.md` | Lệnh pilot ghi `--epochs 6`; thực tế pilot C1 chạy 3 epoch | Pilot dùng 3 epoch (CHANGELOG, 10-04); Stage 1b trở đi dùng 6 |
| `docs/analysis_plan.md` | "bf16 (native T4)" đổi thành "bf16, T4 không có bf16 gốc"; cập nhật nhãn H1–H3; thêm tham chiếu `pivot_v2.md` | Ghi sai về phần cứng; tránh trùng nhãn với H1–H7 (bản 4) |
| `docs/roadmap_v2.md` | 12 seed (200–211, 96 lượt) đổi thành 16 seed (200–215, 128 lượt); δ 0,005 đổi thành 0,01 trong các lệnh; bảng chi phí tính lại | Kế hoạch (cập nhật sau power analysis) dùng N = 16 và δ = 0,01; roadmap viết trước đó |
| `code/analyze_global.py` | Mặc định `--delta_f1` từ 0,005 thành 0,01 | Khớp kế hoạch; tránh chạy nhầm TOST ở δ không khả thi. Độ nhạy 0,005 và 0,0075 chạy bằng cờ riêng |
| `docs/decisions.md` | Giao thức đánh giá: test là chính cho họ xác nhận; thêm mục "Hướng v2"; thêm kết quả Stage 1c; Stage 1d ghi "chưa chạy" | Khớp kế hoạch hiện hành |
| `docs/workflow.md` | `pull_kaggle.sh`, `push_cycle.sh` đổi thành `.ps1` | Tên file thực tế trong `scripts/` |
| `CHANGELOG.md` | Thêm mục "Cycle 2b: đổi hướng đề tài" và ghi chú giải quyết cho mục smoke test | Lưu quyết định |
| `docs/archive/` (mới) | Bản markdown của đề xuất bản 4 và đặc tả bản 2 (công thức giữ nguyên), README cũ | Không mất chi tiết hướng ban đầu |
| `docs/exploratory_results.md` (mới) | Tổng hợp số liệu và chú thích pilot, Stage 1a–1c; cảnh báo các câu nháp quá mạnh trong CHANGELOG | Giữ kết quả quan trọng ở một chỗ |

Không sửa: mã huấn luyện (trừ mặc định `--delta_f1`), dữ liệu, kết quả, các mục lịch sử trong CHANGELOG (trừ ghi chú giải quyết bổ sung ở trên).

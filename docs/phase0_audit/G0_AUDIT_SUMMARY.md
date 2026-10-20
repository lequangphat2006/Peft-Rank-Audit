# Phase 0 Audit: G0 Gate Verification

## Mục tiêu
Kiểm tra xem thiết kế thực nghiệm có đủ năng lực phân biệt 3 giả thuyết lý thuyết (H_k=1, H_muA, H_k=1/2) hay không, trong điều kiện bề mặt loss bị méo (plateau, collapse, độ cong thay đổi theo rank).

## Quy trình kiểm toán
- **Công cụ:** Bộ script `kiem_toan_G0_v3_1` (11 bước).
- **Local:** Chạy với `SEED_OFFSET=0` (hạt giống mặc định).
- **Kaggle:** Chạy với `SEED_OFFSET=1000` (hạt giống mới, 4 procs) để kiểm tra độ ổn định Monte Carlo.

## Kết quả chính (Đồng thuận giữa Local và Kaggle)
Cả hai môi trường đều cho ra cùng một kết luận về các "đường biên" (boundaries):
1. **Bước 09 (Shared Surface):** Pass (Accuracy > 0.9) khi \(c_2 \ge 0.1\).
2. **Bước 11 (Giao thức đồng thuận):** An toàn tuyệt đối (Unsafe = 0%) khi \(c_2 \ge 0.2\). Khi \(c_2 \le 0.1\), giao thức tự động giơ cờ (flag) để từ chối đưa ra kết luận sai.

## Ứng dụng cho Pha 1
- Nếu Pilot 0C cho thấy NLL có \(c_2 \ge 0.2\): Chọn thiết kế Bước 2.0, 5 seeds.
- Nếu NLL có \(c_2 \in [0.1, 0.2]\): Chọn thiết kế Bước \(\sqrt{2}\), 5-8 seeds.
- Nếu NLL có \(c_2 < 0.1\): Chuyển sang nhánh "Go hẹp" (bỏ claim về số mũ).
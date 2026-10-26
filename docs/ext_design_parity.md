# EXT-1 — Bảng đồng nhất thiết kế XLM-R và PhoBERT

Mục đích: cho reviewer thấy chính xác cái gì được giữ giống nhau (để so sánh công bằng) và cái gì **không thể** giống nhau (yếu tố gây nhiễu của so sánh hai mô hình). Điền cột "Xác minh" bằng kết quả preflight / log khi chạy.

## A. Được đồng nhất
| Thành phần | XLM-R (R1) | PhoBERT (B1X) | Xác minh |
|---|---|---|---|
| Phương pháp | LoRA, α = 2r | LoRA, α = 2r | cột `alpha_mode` |
| Module đích | cùng nhóm module (Q, K, V, O, FFN) qua `rankcap.py` | như XLM-R | log run_pilot |
| Số lớp | 12 | 12 | cấu hình mô hình |
| Rank | 1, 4, 8, 16, 32 | 1, 4, 8, 16, 32 | `ext_merge.py` |
| Epoch | 6 (chính) | 6 | cột `epochs` |
| Batch, max_len | 32, 128 | 32, 128 | cột `batch_size`, `max_len` |
| Precision | fp16 | fp16 | cột `precision` |
| Chia dữ liệu | train / calibration / test | cùng | `ext_preflight.py` mục [3] |
| Nhãn test | — | trùng từng dòng với XLM-R | `ext_preflight.py` mục [4] |
| Quy trình LR | lưới 4 LR + mở rộng một bước; 3 seed tune 400–402; hòa 0,005 | giống hệt | `ext_lr_rule.py` |
| Chấm LR | Macro-F1 trên calibration | giống | `--eval_file calibration.csv` |
| Chỉ số | Macro-F1, NLL, F1 neutral, ECE sau chỉnh nhiệt | giống | scripts `ext_*` |
| Số seed chạy chính | 16 (200–215) | 16 (300–315) | `ext_merge.py` |
| Phần mềm | peft 0.21.2, transformers 5.16.1 | giống | `ext_preflight.py` mục [6] |

## B. KHÔNG thể đồng nhất (yếu tố gây nhiễu, phải nêu trong bài)
| Yếu tố | XLM-R | PhoBERT |
|---|---|---|
| Tokenizer | SentencePiece (đa ngôn ngữ) | BPE tiếng Việt (cần văn bản tách từ) |
| Tách từ | không | có (công cụ tách từ tiếng Việt) |
| Dữ liệu tiền huấn luyện | CC-100 đa ngôn ngữ | kho tiếng Việt ~20 GB |
| Từ vựng | ~250 nghìn | ~64 nghìn |
| Số tham số | ~278 triệu (phần lớn là embedding) | ~135 triệu |
| Độ dài chuỗi token cùng một câu | khác nhau | khác nhau (tỉ lệ bị cắt ở max_len 128 phải báo cáo: `ext_preflight.py --tok_stats`) |
| Seed | 200–215 | 300–315 (số seed không có ý nghĩa chung giữa hai mô hình) |

## C. Lưu ý diễn giải
Vì A và B cùng thay đổi khi đổi mô hình, hiệu ứng tương tác mô hình × rank (E1) chỉ cho biết **hai mô hình, như được cấu hình ở đây, phản ứng với rank khác nhau**; nó không cho phép quy cho tokenizer, tách từ, kích thước hay dữ liệu tiền huấn luyện. Nói "phụ thuộc mô hình", không nói "chứng minh phụ thuộc kiến trúc".

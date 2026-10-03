\# PEFT Rank Audit



Kiểm toán giả định nền tảng của rank allocation trong Parameter-Efficient Fine-Tuning (PEFT).



\*\*Câu hỏi trung tâm:\*\* Capacity need của một module có đo được, ổn định và chuyển giao được không?



\*\*Đối tượng kiểm toán:\*\* AdaLoRA, IGU-LoRA, BaRA, StatLoRA, và mọi phương pháp rank allocation dựa trên importance score.



\---



\## Trạng thái



| Giai đoạn | Trạng thái | Ngày |

|---|---|---|

| Code viết xong | ✅ | 2026-10-03 |

| Pytest 9/9 | ✅ | 2026-10-03 |

| Data prepared | ✅ | 2026-10-03 |

| Pilot C1 (125 lượt) | ⏳ | — |

| Phân tích G1 | ⏳ | — |

| Chạy chính | ⏳ | — |

| Bản thảo | ⏳ | — |



\---



\## Câu hỏi nghiên cứu



\- \*\*RQ1a/H1:\*\* Capacity need của một lớp có đo được vượt nhiễu seed không?

\- \*\*RQ1b/H2:\*\* Profile có ổn định qua seed không?

\- \*\*RQ1c/H3:\*\* Tác động của các lớp có cộng tính không?

\- \*\*RQ1d/H4:\*\* Profile có chuyển giao qua task, ngôn ngữ, lượng dữ liệu không?

\- \*\*RQ2a/H5:\*\* Proxy nào dự đoán được profile trong cùng task?

\- \*\*RQ2b/H6:\*\* Mất cân bằng lớp có làm giảm giá trị dự đoán của proxy không?

\- \*\*RQ3/H7:\*\* Phân bổ theo profile (oracle) hoặc proxy tốt nhất có vượt phân bổ đều và ngẫu nhiên không?



Chi tiết: xem `docs/decisions.md`.



\---



\## Cấu trúc repo

code/ Source code (rankcap, run\_pilot, analyze\_pilot, prepare\_vsfc)

data/ Dữ liệu (không push, upload Kaggle Dataset)

notebooks/ Notebook Kaggle

results/ Kết quả pull về từ Kaggle

scripts/ Script pull/merge/push

docs/ Workflow và quyết định thiết kế



\---



\## Dữ liệu



\*\*UIT-VSFC sentiment\*\* (Vietnamese Students' Feedback Corpus)



| Split | Số mẫu | Vai trò |

|---|---|---|

| Train | 12,780 | Huấn luyện |

| Calibration | 1,598 | Chọn LR, Temperature Scaling |

| Test | 1,599 | Đánh giá cuối |



\*\*Phân bố lớp (train):\*\*



| Nhãn | Ý nghĩa | Số mẫu | Tỷ lệ |

|---|---|---|---|

| 0 | Negative | 5,986 | 46.8% |

| 1 | Neutral | 522 | 4.1% |

| 2 | Positive | 6,272 | 49.1% |



\*\*Xử lý mất cân bằng:\*\* Class-weighted Cross-Entropy. Không resampling.



\*\*Near-duplicate leakage (cosine ≥ 0.95):\*\*

\- 3.5% val có near-duplicate trong train

\- 2.6% test có near-duplicate trong train



\*\*Gate G0:\*\* N=1500 khả thi cho ρ ∈ {1, 5, 20}.



Data không push lên GitHub. Upload lên Kaggle Dataset `peft-data`.



\---



\## Môi trường



| Vai trò | Nền tảng | Cấu hình |

|---|---|---|

| Code + analyze | Local | Windows, RTX 3050 4GB, torch CPU-only |

| Train | Kaggle | T4 16GB, torch CUDA |



\*\*Thư viện chính:\*\*

\- torch ≥ 2.1

\- transformers ≥ 4.45

\- peft ≥ 0.12

\- scikit-learn, scipy, statsmodels, pandas



Cài đặt: `pip install -r requirements.txt`



\---



\## Cách chạy



\### Local — prepare data



```bash

python code/prepare\_vsfc.py \\

&#x20; --train data/raw/train.csv \\

&#x20; --val   data/raw/val.csv \\

&#x20; --test  data/raw/test.csv \\

&#x20; --out\_dir data/vsfc\_sent \\

&#x20; --text\_col text --label\_col label


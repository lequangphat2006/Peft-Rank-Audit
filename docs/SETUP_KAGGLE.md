\# Setup Kaggle — checklist cho MỌI session mới



Cập nhật: 2026-10-16. Áp dụng cho mọi notebook Kaggle, mọi tài khoản.

Đọc mục 1 trước khi chạy bất cứ cell nào.



\---



\## 0. Cấu hình notebook (Settings cột phải)



| Mục | Giá trị |

|---|---|

| Accelerator | \*\*GPU T4 ×2\*\* |

| Internet | \*\*On\*\* |

| Persistence | Files only |

| Add Data | `lequangphat260206/peft-data` |



\---



\## 1. BẮT BUỘC — Gỡ torchao trước khi chạy



Kaggle image cài sẵn `torchao 0.10.0`. Peft 0.20.0 \*\*yêu cầu torchao ≥ 0.16\*\* khi import, nên nếu giữ nguyên phiên bản cũ thì mọi lần `get\_peft\_model` sẽ crash với:



```

ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported

```



\*\*Chạy cell này ĐẦU TIÊN ở mỗi session mới:\*\*



```python

import subprocess

r = subprocess.run(\["pip","uninstall","-y","torchao"], capture\_output=True, text=True)

print(r.stdout\[-300:])

\# Xác nhận

r2 = subprocess.run(\["python","-c",

&#x20;   "import importlib.util as u; print('torchao present:', u.find\_spec('torchao') is not None)"],

&#x20;   capture\_output=True, text=True)

print(r2.stdout, r2.stderr)

```



Kỳ vọng: `Successfully uninstalled torchao-0.10.0` và `torchao present: False`.



\*\*Tại sao gỡ mà không nâng cấp?\*\*

\- Peft chỉ cần torchao khi bật quantization (`load\_in\_4bit` / `load\_in\_8bit`). Dự án này không dùng.

\- Nâng cấp torchao nặng (\~500 MB download) và có thể xung đột với torch 2.11.

\- Gỡ 0.10.0 mất \~5 giây và không phá gì khác.



\*\*Lưu ý:\*\* Session Kaggle mới sẽ \*\*cài lại\*\* torchao 0.10.0. Phải chạy lại cell này mỗi lần restart.



\---



\## 2. Kiểm tra môi trường chuẩn



Chạy sau cell gỡ torchao:



```python

import torch, transformers, peft

print(f"torch        : {torch.\_\_version\_\_}")

print(f"transformers : {transformers.\_\_version\_\_}")

print(f"peft         : {peft.\_\_version\_\_}")

print(f"CUDA         : {torch.version.cuda}")

print(f"GPU          : {torch.cuda.get\_device\_name(0)}")

print(f"SM           : {torch.cuda.get\_device\_capability(0)}")

```



\*\*Kỳ vọng (Kaggle 2026-10):\*\*



```

torch        : 2.11.0+cu128

transformers : 5.16.1

peft         : 0.20.0

CUDA         : 12.8

GPU          : Tesla T4

SM           : (7, 5)

```



Nếu peft ≠ 0.20.0 → cài lại: `pip install "peft==0.20.0"`.



\---



\## 3. Clone code



```python

import os, subprocess, shutil

os.chdir("/kaggle/working")                                    # luôn chdir về đây trước

CODE\_DIR = "/kaggle/working/code"

if os.path.exists(CODE\_DIR): shutil.rmtree(CODE\_DIR)

subprocess.run(\["git","clone","https://github.com/lequangphat2006/Peft-Rank-Audit.git", CODE\_DIR],

&#x20;              check=True)

os.chdir(CODE\_DIR)

print(subprocess.run(\["git","log","--oneline","-3"], capture\_output=True, text=True).stdout)

print(subprocess.run(\["git","tag"], capture\_output=True, text=True).stdout)

```



\*\*Lưu ý:\*\* code nằm ở `/kaggle/working/code/code/` (double `code/`).



\- Repo root: `/kaggle/working/code/`

\- Python scripts: `/kaggle/working/code/code/run\_pilot.py`

\- Docs: `/kaggle/working/code/docs/`



Khi gọi `run\_pilot.py` từ subprocess, dùng `cwd="/kaggle/working/code/code"` hoặc đường dẫn đầy đủ.



\---



\## 4. Kiểm tra data mount



```python

import os

DATA\_DIR = "/kaggle/input/datasets/lequangphat260206/peft-data"

assert os.path.exists(DATA\_DIR), f"DATA\_DIR not found: {DATA\_DIR}"

for f in \["train.csv","calibration.csv","test.csv"]:

&#x20;   p = os.path.join(DATA\_DIR, f)

&#x20;   print(f"{f}: {os.path.getsize(p)/1e6:.2f} MB")

```



\*\*Kỳ vọng:\*\* 3 file với kích thước \~1 MB, \~0.13 MB, \~0.13 MB.



\---



\## 5. Lỗi thường gặp và cách xử lý



| Lỗi | Nguyên nhân | Fix |

|---|---|---|

| `fatal: Unable to read current working directory` | CWD trỏ vào thư mục đã xóa | `os.chdir("/kaggle/working")` ở đầu mọi cell |

| `ImportError: torchao 0.10.0...` | Quên gỡ torchao | Chạy lại cell mục 1 |

| `No module named 'rankcap'` | Chưa `sys.path.insert` hoặc clone sai | `sys.path.insert(0, "/kaggle/working/code/code")` |

| `FileNotFoundError: train.csv` | DATA\_DIR sai | Dùng đường dẫn đầy đủ ở mục 4 |

| `CUDA out of memory` | Batch quá lớn | Giảm `--batch\_size 16` hoặc `--max\_len 96` |

| Notebook tự reset giữa chừng | Session hết hạn (12h) | Lưu `/kaggle/working/results` ra Dataset trước khi hết giờ |



\---



\## 6. Template cell 0 — copy vào đầu mọi notebook



```python

\# === CELL 0: SETUP BẮT BUỘC (chạy đầu tiên) ===

import os, subprocess, sys, shutil



\# 1. Gỡ torchao (xung đột với peft 0.20.0)

subprocess.run(\["pip","uninstall","-y","torchao"], capture\_output=True, text=True)



\# 2. Chdir về thư mục hợp lệ

os.chdir("/kaggle/working")



\# 3. Clone code

CODE\_DIR = "/kaggle/working/code"

if os.path.exists(CODE\_DIR): shutil.rmtree(CODE\_DIR)

subprocess.run(\["git","clone","https://github.com/lequangphat2006/Peft-Rank-Audit.git", CODE\_DIR], check=True)

os.chdir(CODE\_DIR)

sys.path.insert(0, os.path.join(CODE\_DIR, "code"))



\# 4. Kiểm tra

import torch, transformers, peft

print(f"torch={torch.\_\_version\_\_} transformers={transformers.\_\_version\_\_} peft={peft.\_\_version\_\_}")

print(f"GPU={torch.cuda.get\_device\_name(0)} SM={torch.cuda.get\_device\_capability(0)}")

print(subprocess.run(\["git","log","--oneline","-1"], capture\_output=True, text=True).stdout.strip())

print("Setup done. Sẵn sàng chạy job.")

```



Chạy cell này xong là môi trường sẵn sàng. Mọi cell tiếp theo có thể gọi `run\_pilot.py` với `cwd="/kaggle/working/code/code"`.



\---



\## 7. Nhiều tài khoản Kaggle — lưu ý phân chia



Khi chạy song song nhiều account, \*\*phải đảm bảo mỗi account chạy đúng shard\*\*, không trùng seed:



| Account | Shard | Seeds | Ghi chú |

|---|---|---|---|

| A | `--shard 0/4` | 200–203 | |

| B | `--shard 1/4` | 204–207 | |

| C | `--shard 2/4` | 208–211 | |

| D | `--shard 3/4` | 212–215 | |



\*\*Quan trọng:\*\*

\- Mỗi account dùng \*\*file output riêng\*\* (`/kaggle/working/global\_gpu\_a.csv`, v.v.)

\- Sau khi xong mỗi account, tải về máy local, gộp bằng `pd.concat` ở local

\- \*\*Không\*\* chạy 2 account với cùng shard — sẽ ra kết quả trùng

\- `run\_id` bao gồm seed, nên gộp không bị collision



\*\*Kiểm tra hạn mức:\*\* mỗi account mới có \~30 GPU-giờ/tuần. 4 account → \~120 GPU-giờ/tuần, đủ chạy Phase 1 (\~28 GPU-giờ) trong 1 tuần.



\---



\## 8. Checklist trước khi bắt đầu chạy thí nghiệm thật



\- \[ ] Đã chạy cell gỡ torchao (mục 1)

\- \[ ] Đã xác nhận môi trường (mục 2): torch 2.11, transformers 5.16, peft 0.20

\- \[ ] Đã clone code và thấy commit hash đúng (mục 3)

\- \[ ] Đã mount data và kiểm tra 3 file (mục 4)

\- \[ ] `analysis\_plan\_v2.md` có tag `analysis-plan-v2.0` (không chỉ draft) trước khi chạy R1

\- \[ ] File output dùng tên riêng cho mỗi shard/account

\- \[ ] Có kế hoạch lưu kết quả ra Dataset trong vòng 12h (session limit)



\---



\## 9. Khi session bị ngắt giữa chừng



Kaggle session tối đa 12 giờ. Nếu job bị cắt:



1\. Vào notebook, \*\*Save Version\*\* để lưu `/kaggle/working` hiện tại

2\. Output mới sẽ nằm ở `/kaggle/working/results/global\_gpu0.csv` — tải về máy

3\. Mở session mới, chạy cell 0 (setup)

4\. Copy file CSV đã chạy một phần vào `/kaggle/working/results/` trước khi chạy lại

5\. Chạy lại cùng lệnh — `run\_pilot.py` đọc file `--out` đã có và \*\*bỏ qua\*\* lượt đã xong (resume theo `run\_id`)



Xem thêm: `docs/workflow.md` mục "Khi Kaggle bị ngắt".


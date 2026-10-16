# Setup Kaggle — checklist cho MỌI session mới

Cập nhật: 2026-10-16. Áp dụng cho mọi notebook Kaggle, mọi tài khoản.

**Đọc mục 1 trước khi chạy bất cứ cell nào.**

OSF Pre-registration: https://osf.io/a2mn5/ · DOI: 10.17605/OSF.IO/A2MN5

---

## 0. Cấu hình notebook (Settings cột phải)

| Mục | Giá trị |
|---|---|
| Accelerator | GPU T4 ×2 |
| Internet | On (BẮT BUỘC — kiểm tra mỗi session mới) |
| Persistence | Files only |
| Add Data | `lequangphat260206/peft-data` |

---

## 1. BẮT BUỘC — Gỡ torchao trước khi chạy

Kaggle image cài sẵn torchao 0.10.0. Peft 0.20.0 yêu cầu torchao ≥ 0.16 khi import, nên nếu giữ nguyên phiên bản cũ thì mọi lần `get_peft_model` sẽ crash với:

```text
ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported
```

Chạy cell này **ĐẦU TIÊN** ở mỗi session mới:

```python
import subprocess

r = subprocess.run(["pip","uninstall","-y","torchao"], capture_output=True, text=True)
print(r.stdout[-300:])
r2 = subprocess.run(["python","-c",
    "import importlib.util as u; print('torchao present:', u.find_spec('torchao') is not None)"],
    capture_output=True, text=True)
print(r2.stdout, r2.stderr)
```

Kỳ vọng: `Successfully uninstalled torchao-0.10.0` và `torchao present: False`.

### Tại sao gỡ mà không nâng cấp?

- Peft chỉ cần torchao khi bật quantization (`load_in_4bit` / `load_in_8bit`). Dự án này không dùng.
- Nâng cấp torchao nặng (~500 MB download) và có thể xung đột với torch 2.11.
- Gỡ 0.10.0 mất ~5 giây và không phá gì khác.

> **Lưu ý:** Session Kaggle mới sẽ cài lại torchao 0.10.0. Phải chạy lại cell này mỗi lần restart.

---

## 2. Kiểm tra môi trường chuẩn

Chạy sau cell gỡ torchao:

```python
import torch, transformers, peft
print(f"torch        : {torch.__version__}")
print(f"transformers : {transformers.__version__}")
print(f"peft         : {peft.__version__}")
print(f"CUDA         : {torch.version.cuda}")
print(f"GPU          : {torch.cuda.get_device_name(0)}")
print(f"SM           : {torch.cuda.get_device_capability(0)}")
print(f"bf16 flag    : {torch.cuda.is_bf16_supported()} (BUG trên T4: luôn True dù SM 7.5)")
```

Kỳ vọng (Kaggle 2026-10):

```text
torch        : 2.11.0+cu128
transformers : 5.16.1
peft         : 0.20.0
CUDA         : 12.8
GPU          : Tesla T4
SM           : (7, 5)
bf16 flag    : True   (nhưng là bug, T4 KHÔNG có bf16 gốc)
```

Nếu peft ≠ 0.20.0 → cài lại: `pip install "peft==0.20.0"`.

---

## 3. Clone code

```python
import os, subprocess, sys, shutil

os.chdir("/kaggle/working")
CODE_DIR = "/kaggle/working/code"
if os.path.exists(CODE_DIR): shutil.rmtree(CODE_DIR)
subprocess.run(["git","clone","https://github.com/lequangphat2006/Peft-Rank-Audit.git", CODE_DIR],
               check=True)
os.chdir(CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, "code"))
print(subprocess.run(["git","log","--oneline","-3"], capture_output=True, text=True).stdout)
print(subprocess.run(["git","tag"], capture_output=True, text=True).stdout)
```

**Lưu ý:** code nằm ở `/kaggle/working/code/code/` (double `code/`).

- Repo root: `/kaggle/working/code/`
- Python scripts: `/kaggle/working/code/code/run_pilot.py`
- Docs: `/kaggle/working/code/docs/`

Khi gọi `run_pilot.py` từ subprocess, dùng `cwd="/kaggle/working/code/code"` hoặc đường dẫn đầy đủ.

Xác nhận tag: phải thấy `analysis-plan-v2.0` và `osf-preregistered-v2.0` trong output `git tag`.

---

## 4. Kiểm tra data mount

```python
import os

DATA_DIR = "/kaggle/input/datasets/lequangphat260206/peft-data"
assert os.path.exists(DATA_DIR), f"DATA_DIR not found: {DATA_DIR}. Add dataset peft-data."
for f in ["train.csv","calibration.csv","test.csv"]:
    p = os.path.join(DATA_DIR, f)
    print(f"{f}: {os.path.getsize(p)/1e6:.2f} MB")
```

Kỳ vọng: 3 file với kích thước ~1 MB, ~0.13 MB, ~0.13 MB.

---

## 5. Lỗi thường gặp và cách xử lý

| Lỗi | Nguyên nhân | Fix |
|---|---|---|
| `fatal: Unable to read current working directory` | CWD trỏ vào thư mục đã xóa | `os.chdir("/kaggle/working")` ở đầu mọi cell |
| `Could not resolve host: github.com` | Internet chưa bật | Bật Internet trong Settings |
| `ImportError: torchao 0.10.0...` | Quên gỡ torchao | Chạy lại cell mục 1 |
| `No module named 'rankcap'` | Chưa `sys.path.insert` hoặc clone sai | `sys.path.insert(0, "/kaggle/working/code/code")` |
| `FileNotFoundError: train.csv` | `DATA_DIR` sai | Dùng đường dẫn đầy đủ ở mục 4 |
| `CUDA out of memory` | Batch quá lớn | Giảm `--batch_size 16` hoặc `--max_len 96` |
| Notebook tự reset giữa chừng | Session hết hạn (12h) | Lưu `/kaggle/working/results` ra Dataset trước khi hết giờ |
| Tag `analysis-plan-v2.0` không thấy | Clone code cũ | `shutil.rmtree` + clone lại |

---

## 6. Template cell 0 — copy vào đầu mọi notebook

```python
# === CELL 0: SETUP BẮT BUỘC (chạy đầu tiên) ===
import os, subprocess, sys, shutil

# 1. Gỡ torchao (xung đột với peft 0.20.0)
subprocess.run(["pip","uninstall","-y","torchao"], capture_output=True, text=True)

# 2. Chdir về thư mục hợp lệ
os.chdir("/kaggle/working")

# 3. Clone code
CODE_DIR = "/kaggle/working/code"
if os.path.exists(CODE_DIR): shutil.rmtree(CODE_DIR)
subprocess.run(["git","clone","https://github.com/lequangphat2006/Peft-Rank-Audit.git", CODE_DIR], check=True)
os.chdir(CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, "code"))

# 4. Kiểm tra
import torch, transformers, peft
print(f"torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__}")
print(f"GPU={torch.cuda.get_device_name(0)} SM={torch.cuda.get_device_capability(0)}")
print(subprocess.run(["git","log","--oneline","-1"], capture_output=True, text=True).stdout.strip())
print("Tags:", subprocess.run(["git","tag"], capture_output=True, text=True).stdout.split())
print("Setup done. Sẵn sàng chạy job.")
```

Chạy cell này xong là môi trường sẵn sàng. Mọi cell tiếp theo có thể gọi `run_pilot.py` với `cwd="/kaggle/working/code/code"`.

---

## 7. Nhiều tài khoản Kaggle — lưu ý phân chia

Khi chạy song song nhiều account, phải đảm bảo mỗi account chạy đúng shard, không trùng seed:

| Account | Shard | Seeds | Ghi chú |
|---|---|---|---|
| A | `--shard 0/4` | 200–203 | |
| B | `--shard 1/4` | 204–207 | |
| C | `--shard 2/4` | 208–211 | |
| D | `--shard 3/4` | 212–215 | |

**Quan trọng:**

- Mỗi account dùng file output riêng (`/kaggle/working/global_gpu_a.csv`, v.v.)
- Sau khi xong mỗi account, tải về máy local, gộp bằng `pd.concat` ở local
- Không chạy 2 account với cùng shard — sẽ ra kết quả trùng
- `run_id` bao gồm seed, nên gộp không bị collision

Kiểm tra hạn mức: mỗi account mới có ~30 GPU-giờ/tuần. 4 account → ~120 GPU-giờ/tuần, đủ chạy Phase 1 (~28 GPU-giờ) trong 1 tuần.

---

## 8. Checklist trước khi bắt đầu chạy thí nghiệm thật

- [ ] Đã chạy cell gỡ torchao (mục 1)
- [ ] Đã xác nhận môi trường (mục 2): torch 2.11, transformers 5.16, peft 0.20
- [ ] Đã clone code và thấy commit hash đúng (mục 3)
- [ ] Đã mount data và kiểm tra 3 file (mục 4)
- [ ] `analysis_plan_v2.0.md` có tag `analysis-plan-v2.0` (không chỉ draft) trước khi chạy R1
- [ ] File output dùng tên riêng cho mỗi shard/account
- [ ] Có kế hoạch lưu kết quả ra Dataset trong vòng 12h (session limit)
- [x] OSF pre-registration đã public: https://osf.io/a2mn5/ (DOI: 10.17605/OSF.IO/A2MN5)

---

## 9. Khi session bị ngắt giữa chừng

Kaggle session tối đa 12 giờ. Nếu job bị cắt:

1. Vào notebook, **Save Version** để lưu `/kaggle/working` hiện tại
2. Output mới sẽ nằm ở `/kaggle/working/results/...` — tải về máy
3. Mở session mới, chạy cell 0 (setup)
4. Copy file CSV đã chạy một phần vào `/kaggle/working/results/` trước khi chạy lại
5. Chạy lại cùng lệnh — `run_pilot.py` đọc file `--out` đã có và bỏ qua lượt đã xong (resume theo `run_id`)

Xem thêm: `docs/workflow.md` mục "Khi Kaggle bị ngắt".

---

## 10. Các cell chạy chuẩn

### 10.1. Tune LR theo rank (Phase 1, seed 400–402)

```python
import os, subprocess, time
import pandas as pd
os.chdir("/kaggle/working")

RUNNER   = "/kaggle/working/code/code/run_pilot.py"
WORKDIR  = "/kaggle/working/code/code"
DATA_DIR = "/kaggle/input/datasets/lequangphat260206/peft-data"
OUT_DIR  = "/kaggle/working/results/tuning"
os.makedirs(OUT_DIR, exist_ok=True)

base_args = [
    "python", RUNNER,
    "--data_dir", DATA_DIR,
    "--model", "FacebookAI/xlm-roberta-base",
    "--cell", "TUNE",
    "--sweep_global",
    "--ranks", "1", "4", "8", "16", "32",
    "--lr_list", "1e-4", "2e-4", "5e-4", "1e-3",
    "--epochs_list", "6",
    "--seeds", "400", "401", "402",
    "--precision", "fp16",
    "--eval_file", "calibration.csv",
    "--epoch_eval", "none",
]

procs = []
for gpu, shard in [(0, "0/2"), (1, "1/2")]:
    out = f"{OUT_DIR}/tuning_gpu{gpu}.csv"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    log = open(f"{OUT_DIR}/tuning_gpu{gpu}.log", "w")
    p = subprocess.Popen(base_args + ["--shard", shard, "--out", out],
                         cwd=WORKDIR, env=env, stdout=log, stderr=subprocess.STDOUT)
    procs.append((gpu, p, out, log))
    print(f"Launched tune GPU{gpu}")

t0 = time.time()
for gpu, p, out, log in procs:
    p.wait(); log.close()
    print(f"GPU{gpu} done, exit={p.returncode}, elapsed={time.time()-t0:.0f}s")

dfs = [pd.read_csv(f"{OUT_DIR}/tuning_gpu{g}.csv") for g in (0, 1)]
merged = pd.concat(dfs, ignore_index=True)
merged.to_csv(f"{OUT_DIR}/tuning.csv", index=False)
print(f"\ntuning.csv: {len(merged)} rows (kỳ vọng 60)")
```

### 10.2. R1 xác nhận (Phase 2, seed 200–215)

```python
import os, subprocess, time
import pandas as pd
os.chdir("/kaggle/working")

RUNNER   = "/kaggle/working/code/code/run_pilot.py"
WORKDIR  = "/kaggle/working/code/code"
DATA_DIR = "/kaggle/input/datasets/lequangphat260206/peft-data"
OUT_DIR  = "/kaggle/working/results/global"
os.makedirs(OUT_DIR, exist_ok=True)

base_args = [
    "python", RUNNER,
    "--data_dir", DATA_DIR,
    "--model", "FacebookAI/xlm-roberta-base",
    "--cell", "R1",
    "--sweep_global",
    "--ranks", "1", "4", "8", "16", "32",
    "--epochs_list", "3", "6",
    "--seeds", "200", "201", "202", "203", "204", "205", "206", "207",
    "--lr", "2e-4",
    "--precision", "fp16",
    "--eval_file", "test.csv",
    "--cal_file", "calibration.csv",
    "--epoch_eval", "cal",
]

procs = []
for gpu, shard in [(0, "0/2"), (1, "1/2")]:
    out = f"{OUT_DIR}/global_gpu{gpu}.csv"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    log = open(f"{OUT_DIR}/global_gpu{gpu}.log", "w")
    p = subprocess.Popen(base_args + ["--shard", shard, "--out", out],
                         cwd=WORKDIR, env=env, stdout=log, stderr=subprocess.STDOUT)
    procs.append((gpu, p, out, log))
    print(f"Launched R1 GPU{gpu}")

t0 = time.time()
for gpu, p, out, log in procs:
    p.wait(); log.close()
    print(f"GPU{gpu} done, exit={p.returncode}, elapsed={time.time()-t0:.0f}s")

dfs = [pd.read_csv(f"{OUT_DIR}/global_gpu{g}.csv") for g in (0, 1)]
merged = pd.concat(dfs, ignore_index=True)
merged.to_csv(f"{OUT_DIR}/global_merged.csv", index=False)
print(f"\nglobal_merged.csv: {len(merged)} rows (kỳ vọng 80 cho đợt 1)")
```

### 10.3. Phân tích H1–H3

```python
import subprocess, os
os.chdir("/kaggle/working/code")

r = subprocess.run(["python", "code/analyze_global.py",
                    "/kaggle/working/results/global/global_merged.csv",
                    "--primary_epochs", "6",
                    "--delta_f1", "0.01",
                    "--lr_decision", "/kaggle/working/results/tuning/tuning_decision.json",
                    "--fixed_lr", "2e-4",
                    "--logits_dir", "/kaggle/working/results/global/logits",
                    "--out", "/kaggle/working/results/analysis/global_v1.json"],
                   capture_output=True, text=True)
print(r.stdout)
if r.stderr: print("STDERR:", r.stderr)
```

### 10.4. Save Version cuối session

1. Bấm **Save Version** góc phải
2. Chọn **Save & Run All (Commit)**
3. Vào tab **Output** → tải file kết quả cần thiết

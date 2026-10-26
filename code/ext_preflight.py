"""ext_preflight.py - kiểm tra TRƯỚC khi tốn giờ GPU cho EXT-1.

Bắt các lỗi âm thầm đã gặp hoặc có thể gặp:
  [1] LR map thiếu rank  -> run_pilot.py bỏ qua rank đó mà không báo lỗi (rank 8 của PhoBERT!).
  [2] Seed tune trùng seed chạy chính (V4 cũ dùng 300-302 cho cả hai).
  [3] Dữ liệu PhoBERT: đủ 3 file, đúng số dòng, nhãn thuộc {0,1,2}, văn bản đã tách từ (có '_').
  [4] Nhãn test của PhoBERT khớp từng dòng với nhãn test của XLM-R -> bootstrap ghép cặp theo mẫu test mới hợp lệ.
  [5] (tùy chọn) Tỉ lệ câu bị cắt ở max_len của từng tokenizer (cần mạng HuggingFace).
  [6] Phiên bản thư viện so với bản đã dùng cho R1.
  [7] (tùy chọn) --dry_run của run_pilot.py: số lượt và số lượt đã xong (cần torch).

Ví dụ:
  python code/ext_preflight.py --lr_map results/ext/phobert/phobert_lr_map_v2.json \
      --data_phobert /kaggle/input/peft-data/vsfc_sent_phobert --data_xlmr /kaggle/input/peft-data/vsfc_sent \
      --final_seeds 300-315 --tune_seeds 400-402 --tok_stats --dry_run_out results/ext/phobert/B1_ext_gpu0.csv
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_common import read_csv, load_lr_map  # noqa: E402
from ext_merge import parse_seeds  # noqa: E402

EXPECT_ROWS = {"train.csv": 12780, "calibration.csv": 1598, "test.csv": 1599}
PINS = {"peft": "0.21.2", "transformers": "5.16.1"}

_fail = 0


def ok(msg):
    print(f"  [OK]   {msg}")


def warn(msg):
    print(f"  [CHÚ Ý] {msg}")


def bad(msg):
    global _fail
    _fail += 1
    print(f"  [LỖI]  {msg}")


def check_lr_map(path, ranks):
    print("[1] LR map")
    m = load_lr_map(path)
    miss = [r for r in ranks if r not in m]
    if miss:
        bad(f"LR map thiếu rank {miss} - run_pilot.py sẽ BỎ QUA các rank này một cách im lặng")
    else:
        ok("LR map có đủ rank: " + ", ".join(f"{r}->{m[r]:g}" for r in ranks))
    extra = [r for r in m if r not in ranks]
    if extra:
        warn(f"LR map có rank thừa {extra} (không sao, sẽ không được dùng)")
    return m


def check_seeds(final, tune):
    print("[2] Seed")
    inter = sorted(set(final) & set(tune))
    if inter:
        bad(f"seed tune trùng seed chạy chính: {inter}")
    else:
        ok(f"seed tune {tune[0]}..{tune[-1]} tách khỏi seed chạy chính {final[0]}..{final[-1]} ({len(final)} seed)")
    if len(final) != 16:
        warn(f"số seed chạy chính = {len(final)} (đề xuất của cô: 16)")


def seg_fraction(texts):
    return float(np.mean([("_" in str(t)) for t in texts]))


def check_data(dp, dx):
    print("[3] Dữ liệu")
    tests = {}
    for tag, d in (("PhoBERT", dp), ("XLM-R", dx)):
        if not d:
            continue
        for fn, n in EXPECT_ROWS.items():
            p = os.path.join(d, fn)
            if not os.path.exists(p):
                bad(f"{tag}: thiếu {p}")
                continue
            df = read_csv(p)
            if list(df.columns[:2]) != ["text", "label"]:
                bad(f"{tag}/{fn}: cột {list(df.columns)} (cần text,label)")
                continue
            if len(df) != n:
                bad(f"{tag}/{fn}: {len(df)} dòng (mong đợi {n})")
            if not set(df["label"].unique()) <= {0, 1, 2}:
                bad(f"{tag}/{fn}: nhãn lạ {sorted(df['label'].unique())}")
            if fn == "test.csv":
                tests[tag] = df
            if tag == "PhoBERT" and fn == "train.csv":
                f = seg_fraction(df["text"].head(2000))
                (ok if f > 0.5 else bad)(f"PhoBERT: {f:.0%} câu có dấu '_' (văn bản đã tách từ?)")
            if tag == "XLM-R" and fn == "train.csv":
                f = seg_fraction(df["text"].head(2000))
                (ok if f < 0.2 else warn)(f"XLM-R: {f:.0%} câu có dấu '_' (nên là văn bản chưa tách từ)")
    print("[4] Khớp nhãn test giữa hai mô hình (điều kiện của bootstrap ghép cặp)")
    if len(tests) == 2:
        a, b = tests["PhoBERT"]["label"].to_numpy(), tests["XLM-R"]["label"].to_numpy()
        if len(a) == len(b) and (a == b).all():
            ok(f"nhãn test trùng khớp từng dòng ({len(a)} mẫu)")
        else:
            bad("nhãn test KHÁC nhau giữa hai thư mục -> không thể ghép cặp theo mẫu test")
    else:
        warn("bỏ qua: cần cả --data_phobert và --data_xlmr")


def check_tokens(dp, dx, max_len=128):
    print("[5] Tỉ lệ cắt ở max_len")
    try:
        from transformers import AutoTokenizer
        for tag, d, name in (("PhoBERT", dp, "vinai/phobert-base"), ("XLM-R", dx, "FacebookAI/xlm-roberta-base")):
            if not d:
                continue
            tok = AutoTokenizer.from_pretrained(name)
            texts = read_csv(os.path.join(d, "train.csv"))["text"].astype(str).tolist()
            L = np.array([len(x) for x in tok(texts, truncation=False)["input_ids"]])
            print(f"      {tag}: độ dài token trung vị {np.median(L):.0f}, p95 {np.percentile(L, 95):.0f}, "
                  f"tỉ lệ > {max_len}: {np.mean(L > max_len):.2%}")
    except Exception as e:  # noqa: BLE001
        warn(f"không tải được tokenizer ({type(e).__name__}); bỏ qua mục này")


def check_versions():
    print("[6] Phiên bản thư viện")
    try:
        import peft, transformers, torch
        have = {"peft": peft.__version__, "transformers": transformers.__version__}
        for k, v in PINS.items():
            (ok if have[k] == v else warn)(f"{k} {have[k]} (R1 dùng {v})")
        print(f"      torch {torch.__version__}")
    except Exception as e:  # noqa: BLE001
        warn(f"không import được thư viện ({type(e).__name__}); bỏ qua")


def dry_run(args_out, data_phobert, lr_map, seeds, ranks, model, tune=False):
    print("[7] run_pilot.py --dry_run")
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, os.path.join(here, "run_pilot.py"), "--data_dir", data_phobert or ".", "--model", model,
           "--out", args_out, "--cell", "B1X", "--sweep_global", "--ranks", *map(str, ranks),
           "--epochs_list", "6", "--seeds", *map(str, seeds), "--lr_map", lr_map, "--precision", "fp16", "--dry_run"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        m = re.search(r"(\d+) jobs in this shard, (\d+) already done, (\d+) to run", r.stdout)
        if not m:
            warn("không đọc được kết quả dry_run: " + (r.stderr.strip().splitlines() or ["?"])[-1])
            return
        n, done, todo = map(int, m.groups())
        want = len(ranks) * len(seeds)
        (ok if n == want else bad)(f"{n} lượt trong danh sách (mong đợi {want}); đã xong {done}, còn {todo}")
    except Exception as e:  # noqa: BLE001
        warn(f"không chạy được dry_run ({type(e).__name__})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lr_map", required=True)
    ap.add_argument("--ranks", type=int, nargs="*", default=[1, 4, 8, 16, 32])
    ap.add_argument("--final_seeds", nargs="*", default=["300-315"])
    ap.add_argument("--tune_seeds", nargs="*", default=["400-402"])
    ap.add_argument("--data_phobert", default=None)
    ap.add_argument("--data_xlmr", default=None)
    ap.add_argument("--tok_stats", action="store_true")
    ap.add_argument("--dry_run_out", default=None)
    ap.add_argument("--model", default="vinai/phobert-base")
    a = ap.parse_args()

    final, tune = parse_seeds(a.final_seeds), parse_seeds(a.tune_seeds)
    check_lr_map(a.lr_map, a.ranks)
    check_seeds(final, tune)
    if a.data_phobert or a.data_xlmr:
        check_data(a.data_phobert, a.data_xlmr)
        if a.tok_stats:
            check_tokens(a.data_phobert, a.data_xlmr)
    check_versions()
    if a.dry_run_out:
        dry_run(a.dry_run_out, a.data_phobert, a.lr_map, final, a.ranks, a.model)
    print(f"\nKết quả: {_fail} lỗi")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()

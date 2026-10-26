"""ext_merge.py - gộp các shard CSV của run_pilot.py, khử trùng theo run_id, kiểm tra đủ lưới, ghi manifest.

Quy trình ghép shard giữ như docs/analysis_plan_v2.0.md mục 9 (2026-10-17): dedupe theo `run_id`
(giữ dòng đầu tiên, báo nếu hai dòng trùng run_id có số liệu khác nhau), rồi kiểm tra số dòng.

Ví dụ (chạy chính 5 rank x 16 seed, LR theo rank):
  python code/ext_merge.py "results/ext/phobert/B1_16seed_gpu*.csv" results/v4_phobert/B1_final_merged.csv \
      --out results/ext/phobert/B1_ext_merged.csv --lr_map results/ext/phobert/phobert_lr_map_v2.json \
      --ranks 1 4 8 16 32 --seeds 300-315 --epochs 6 --cell B1

Thoát mã 1 nếu thiếu lượt (trừ khi --allow_incomplete).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_common import read_csv, load_lr_map, sha256_file  # noqa: E402


def parse_seeds(tokens):
    out = []
    for t in tokens:
        if "-" in t:
            a, b = t.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(t))
    return out


def merge(paths, tol=1e-6):
    frames, src = [], []
    for p in paths:
        d = read_csv(p)
        d["_src"] = os.path.basename(p)
        frames.append(d)
        src.append((p, len(d)))
    if not frames:
        raise SystemExit("Không có file đầu vào.")
    df = pd.concat(frames, ignore_index=True)
    conflicts = []
    dup = df[df.duplicated("run_id", keep=False)]
    for rid, g in dup.groupby("run_id"):
        vals = g[["macro_f1", "nll"]].astype(float).to_numpy()
        if np.abs(vals - vals[0]).max() > tol:
            conflicts.append(rid)
    n_dup = int(df.duplicated("run_id").sum())
    df = df.drop_duplicates("run_id", keep="first").drop(columns=["_src"])
    return df, src, n_dup, conflicts


def expected_grid(lr_map, ranks, seeds, epochs):
    return {(s, r, e): lr_map[r] for s in seeds for r in ranks for e in epochs if r in lr_map}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="đường dẫn hoặc glob (đặt trong dấu nháy)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr_map", default=None, help="nếu có: kiểm tra đủ (seed, rank, epoch) tại LR của map")
    ap.add_argument("--ranks", type=int, nargs="*", default=[1, 4, 8, 16, 32])
    ap.add_argument("--seeds", nargs="*", default=["300-315"])
    ap.add_argument("--epochs", type=int, nargs="*", default=[6])
    ap.add_argument("--cell", default=None, help="giữ lại đúng cell này (vd B1)")
    ap.add_argument("--allow_incomplete", action="store_true")
    a = ap.parse_args()

    paths = []
    for pat in a.inputs:
        paths += sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else [])
    df, src, n_dup, conflicts = merge(paths)
    if a.cell:
        df = df[df["cell"] == a.cell].copy()

    print("Đầu vào:")
    for p, n in src:
        print(f"  {p}: {n} dòng")
    print(f"Sau gộp: {len(df)} lượt duy nhất (loại {n_dup} dòng trùng run_id)")
    ok = True
    if conflicts:
        ok = False
        print(f"LỖI: {len(conflicts)} run_id trùng nhưng số liệu khác nhau, ví dụ {conflicts[:3]}")

    meta = {}
    for col in ("precision", "git", "batch_size", "max_len", "alpha_mode", "class_weight"):
        if col in df.columns:
            meta[col] = sorted(map(str, df[col].dropna().unique()))
    for col, expect in (("precision", ["fp16"]), ("batch_size", ["32"]), ("max_len", ["128"]),
                        ("alpha_mode", None)):
        if col in meta and expect and meta[col] != expect:
            print(f"CẢNH BÁO: {col} = {meta[col]} (mong đợi {expect}); không trộn cấu hình trong một so sánh.")
    if "git" in meta and len(meta["git"]) > 1:
        print(f"CẢNH BÁO: nhiều commit git trong cùng tập kết quả: {meta['git']}")

    missing = []
    if a.lr_map:
        lr_map = load_lr_map(a.lr_map)
        lost = [r for r in a.ranks if r not in lr_map]
        if lost:
            print(f"LỖI: LR map thiếu rank {lost}")
            ok = False
        exp = expected_grid(lr_map, a.ranks, parse_seeds(a.seeds), a.epochs)
        have = {(int(r.seed), int(r["rank"]), int(r.epochs), round(float(r.lr), 10)) for _, r in df.iterrows()}
        for (s, r, e), lr in exp.items():
            if (s, r, e, round(lr, 10)) not in have:
                missing.append((s, r, e, lr))
        print(f"Lưới mong đợi: {len(exp)} lượt; thiếu {len(missing)}")
        if missing:
            ok = False
            for m in missing[:20]:
                print(f"  thiếu seed={m[0]} rank={m[1]} epochs={m[2]} lr={m[3]:g}")
            if len(missing) > 20:
                print(f"  ... và {len(missing) - 20} lượt nữa")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    manifest = dict(out=a.out, n_runs=int(len(df)), sources=[p for p, _ in src], n_duplicates_dropped=n_dup,
                    conflicts=conflicts, missing=[list(map(float, m)) for m in missing], meta=meta,
                    sha256=sha256_file(a.out), complete=bool(ok))
    json.dump(manifest, open(a.out + ".manifest.json", "w", encoding="utf-8"), indent=1)
    print(f"Đã ghi {a.out} và {a.out}.manifest.json | complete={ok}")
    if not ok and not a.allow_incomplete:
        sys.exit(1)


if __name__ == "__main__":
    main()

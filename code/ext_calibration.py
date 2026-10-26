"""ext_calibration.py - hiệu chuẩn (temperature scaling + ECE) cho cả hai mô hình từ logits đã lưu.

Dự án nói về "khả năng phân biệt VÀ hiệu chuẩn". analyze_global.py đã có temperature_scale cho R1
(thăm dò); script này dùng lại đúng hàm đó cho PhoBERT và XLM-R rồi so sánh hai mô hình.
T được ước lượng trên calibration.csv, đánh giá trên test (không rò rỉ).

Đầu ra: bảng theo (mô hình, rank): T, NLL sau chỉnh nhiệt, ECE (15 bin); tổng hiệu ứng rank của từng seed
cho NLL_ts và ECE, và hiệu giữa hai mô hình (Welch). Tất cả là THĂM DÒ (không thuộc họ Holm).

  python code/ext_calibration.py \
    --csv_p results/ext/phobert/B1_ext_merged.csv --lr_map_p results/ext/phobert/phobert_lr_map_v2.json \
    --logits_p results/ext/phobert/logits \
    --csv_x results/global/global_merged.csv --lr_map_x results/tuning/tuning_decision.json \
    --logits_x results/global/logits --out results/ext/phobert/calibration.json
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_common import read_csv, load_lr_map, select_branch, per_seed_effects, welch  # noqa: E402
from analyze_global import temperature_scale  # noqa: E402


def summarize(csv, lr_map, logits_dir, ranks, epochs=6):
    df = read_csv(csv)
    df = select_branch(df[(df["epochs"] == epochs) & df["rank"].isin(ranks)], load_lr_map(lr_map))
    ts = temperature_scale(logits_dir, df[["run_id", "seed", "rank"]])
    if ts.empty:
        raise SystemExit(f"không tìm thấy logits trong {logits_dir}")
    return df.merge(ts, on="run_id")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for m in ("p", "x"):
        ap.add_argument(f"--csv_{m}"); ap.add_argument(f"--lr_map_{m}"); ap.add_argument(f"--logits_{m}")
    ap.add_argument("--ranks", type=int, nargs="*", default=[1, 4, 8, 16, 32])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    res, eff = {}, {}
    for tag, name in (("p", "PhoBERT"), ("x", "XLM-R")):
        if not getattr(a, f"csv_{tag}"):
            continue
        m = summarize(getattr(a, f"csv_{tag}"), getattr(a, f"lr_map_{tag}"), getattr(a, f"logits_{tag}"), a.ranks, a.epochs)
        tab = m.groupby("rank")[["T", "nll", "nll_ts", "ece_ts", "macro_f1"]].mean().round(4)
        print(f"\n=== {name}: nhiệt độ T, NLL, NLL sau chỉnh nhiệt, ECE (n={m['seed'].nunique()} seed) ===")
        print(tab.to_string())
        res[name] = tab.reset_index().to_dict(orient="records")
        eff[name] = {y: per_seed_effects(m, y, a.ranks)[0] for y in ("nll_ts", "ece_ts")}
    if len(eff) == 2:
        print("\n=== Hiệu giữa hai mô hình của tổng hiệu ứng rank (PhoBERT - XLM-R) ===")
        res["difference"] = {}
        for y in ("nll_ts", "ece_ts"):
            w = welch(eff["PhoBERT"][y]["total"], eff["XLM-R"][y]["total"])
            res["difference"][y] = w
            print(f"{y}: {w['diff']:+.4f} CI95 [{w['ci95'][0]:+.4f},{w['ci95'][1]:+.4f}] p={w['p']:.4g}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, default=float)


if __name__ == "__main__":
    main()

"""Chọn LR theo rank từ kết quả tune (chỉ dùng calibration).

Quy tắc (docs/analysis_plan_v2.md mục 5):
- Với mỗi rank, chọn LR có mean Macro-F1 calibration cao nhất (mean qua 3 seed).
- Ngưỡng hòa: 0.005. Khi hòa, chọn LR nhỏ hơn.
- LR tốt nhì: LR khác giá trị với LR đã chọn, có mean F1 cao thứ hai (kể cả khi hòa).

Input : CSV từ run_pilot.py với --eval_file calibration.csv (cột macro_f1 = F1 cal)
Output: results/tuning/tuning_decision.json + bảng in ra

Usage:
  python code/analyze_tuning.py results/tuning/tuning.csv --out results/tuning/tuning_decision.json
"""
import argparse, json, os
import numpy as np
import pandas as pd

LR_THRESH = 0.005


def pick_best_lr(f1_by_lr, thresh=LR_THRESH):
    """f1_by_lr: dict {lr: mean_f1}. Trả (best_lr, second_best_lr, sorted_list giảm dần)."""
    items = sorted(f1_by_lr.items(), key=lambda kv: kv[1], reverse=True)
    if not items:
        return None, None, []
    top_f1 = items[0][1]
    # Tập ứng viên trong ngưỡng hòa với top
    ties = [lr for lr, f1 in items if top_f1 - f1 < thresh]
    best_lr = min(ties)  # khi hòa, chọn LR nhỏ hơn
    others = [(lr, f1) for lr, f1 in items if lr != best_lr]
    second_best_lr = others[0][0] if others else None
    return best_lr, second_best_lr, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--f1_col", default="macro_f1",
                    help="cột dùng để chọn LR. Mặc định macro_f1 vì --eval_file calibration.csv. "
                         "Nếu tune chạy với --eval_file test.csv --cal_file calibration.csv, "
                         "dùng --f1_col cal_macro_f1.")
    ap.add_argument("--out", default="results/tuning/tuning_decision.json")
    ap.add_argument("--thresh", type=float, default=LR_THRESH)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    # Loại row baseline/không phải global nếu có
    if "config" in df.columns:
        df = df[df["config"] == "global"].copy()

    print(f"Loaded {len(df)} runs | ranks={sorted(df['rank'].unique())} "
          f"| lrs={sorted(df['lr'].unique())} | seeds={sorted(df['seed'].unique())}")
    assert a.f1_col in df.columns, f"cột {a.f1_col} không có. Có: {list(df.columns)}"

    decision = {}
    for rank, g in df.groupby("rank"):
        f1_by_lr = g.groupby("lr")[a.f1_col].mean().to_dict()
        n_seeds_by_lr = g.groupby("lr")["seed"].nunique().to_dict()
        best, second, sorted_items = pick_best_lr(f1_by_lr, a.thresh)
        decision[int(rank)] = {
            "best_lr": float(best) if best is not None else None,
            "second_best_lr": float(second) if second is not None else None,
            "threshold": a.thresh,
            "table": [{"lr": float(lr),
                       "mean_f1": float(f1),
                       "n_seeds": int(n_seeds_by_lr.get(lr, 0))}
                      for lr, f1 in sorted_items],
        }

    print("\n=== Chọn LR theo rank (mean Macro-F1 trên calibration) ===")
    print(f"{'rank':>5} | {'best_lr':>10} | {'second_best_lr':>15} | tất cả (lr=mean_f1, n_seeds)")
    print("-" * 100)
    for rank, d in decision.items():
        alls = "  ".join(f"{t['lr']:g}={t['mean_f1']:.4f}(n={t['n_seeds']})" for t in d["table"])
        sb = f"{d['second_best_lr']:g}" if d["second_best_lr"] is not None else "-"
        print(f"{rank:>5} | {d['best_lr']:>10g} | {sb:>15} | {alls}")

    # Cảnh báo nếu thiếu seed
    warn = False
    for rank, d in decision.items():
        for t in d["table"]:
            if t["n_seeds"] < 3:
                print(f"WARNING: rank={rank}, lr={t['lr']:g}, chỉ {t['n_seeds']} seed (kỳ vọng 3)")
                warn = True
    if not warn:
        print("\nOK: mỗi (rank, lr) có đủ 3 seed.")

    # Cổng A: LR tối ưu có dịch theo rank không?
    best_lrs = {d["best_lr"] for d in decision.values()}
    print()
    if len(best_lrs) > 1:
        print(f"=== Cổng A: LR tối ưu KHÁC nhau giữa các rank ({sorted(best_lrs)}) "
              f"-> dùng LR theo rank cho Phase 2 (LR per rank).")
    else:
        print(f"=== Cổng A: LR tối ưu GIỐNG nhau ({sorted(best_lrs)[0]:g}) "
              f"-> có thể dùng LR chung.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(decision, open(a.out, "w"), indent=1)
    print(f"\nĐã lưu: {a.out}")


# --- UTF-8 stdout/stderr (Windows cp1252 fix; idempotent on Linux) ---
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass
# --- end fix ---

if __name__ == "__main__":
    main()
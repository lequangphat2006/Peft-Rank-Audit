"""ext_lr_rule.py - áp dụng ĐÚNG quy tắc chọn LR của XLM-R cho PhoBERT, kể cả mở rộng lưới một bước.

Quy tắc (docs/analysis_plan_v2.0.md mục 5 và mục 9, dòng 2026-10-17):
  1. Lưới cơ sở {1e-4, 2e-4, 5e-4, 1e-3}, 3 seed tune (400-402), 6 epoch, chấm trên calibration.
  2. Mỗi rank: LR có mean Macro-F1 calibration cao nhất; hòa trong ngưỡng 0,005 -> chọn LR nhỏ hơn.
  3. Nếu LR được chọn nằm ở BIÊN lưới cơ sở thì thêm đúng MỘT bước ra ngoài biên
     (5e-5 ở biên dưới, 2e-3 ở biên trên) với cùng 3 seed; chọn lại; KHÔNG mở rộng thêm.
  4. Lượt sập (F1 calibration < 0,25) được giữ trong trung bình và báo cáo tỉ lệ sập theo (rank, LR).

Hai chế độ:
  # (a) sau vòng tune cơ sở: in ra rank nào cần mở rộng và lệnh chạy tương ứng
  python code/ext_lr_rule.py results/ext/phobert/tune_base.csv --stage base

  # (b) sau khi đã có dữ liệu mở rộng (gộp chung 1 CSV): chốt LR map
  python code/ext_lr_rule.py results/ext/phobert/tune_all.csv --stage final \
      --out results/ext/phobert/tuning_decision.json --flat_out results/ext/phobert/phobert_lr_map_v2.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_common import read_csv, CRASH_F1  # noqa: E402
from analyze_tuning import pick_best_lr, LR_THRESH  # noqa: E402

BASE_GRID = [1e-4, 2e-4, 5e-4, 1e-3]
EXT_LOW, EXT_HIGH = 5e-5, 2e-3
TUNE_SEEDS = [400, 401, 402]


def _in(x, lst):
    return any(np.isclose(x, y) for y in lst)


def decide(df, f1_col="macro_f1", thresh=LR_THRESH, stage="final"):
    """Trả về (decision, needs_ext).

    decision[rank] gồm best_lr, second_best_lr, bảng LR, tỉ lệ sập, cờ mở rộng.
    needs_ext[rank] = 'low' | 'high' khi cần chạy thêm LR ngoài biên (chỉ ở stage base).
    """
    d = df[df["config"] == "global"] if "config" in df.columns else df
    decision, needs_ext = {}, {}
    for rank, g in d.groupby("rank"):
        g = g.copy()
        g["crash"] = g[f1_col].astype(float) < CRASH_F1
        by_lr = g.groupby("lr")[f1_col].mean().to_dict()
        nseed = g.groupby("lr")["seed"].nunique().to_dict()
        crash = g.groupby("lr")["crash"].mean().to_dict()

        base_only = {lr: v for lr, v in by_lr.items() if _in(lr, BASE_GRID)}
        best_base, _, _ = pick_best_lr(base_only, thresh)
        edge = None
        if best_base is not None and np.isclose(best_base, min(BASE_GRID)):
            edge = "low"
        elif best_base is not None and np.isclose(best_base, max(BASE_GRID)):
            edge = "high"

        have_ext = {"low": any(np.isclose(lr, EXT_LOW) for lr in by_lr),
                    "high": any(np.isclose(lr, EXT_HIGH) for lr in by_lr)}
        if edge and not have_ext[edge] and stage == "base":
            needs_ext[int(rank)] = edge

        # Ở stage final chỉ xét LR ngoài biên ở phía mà biên cơ sở thực sự là LR chọn (đúng một bước).
        allowed = list(base_only)
        if edge == "low" and have_ext["low"]:
            allowed += [lr for lr in by_lr if np.isclose(lr, EXT_LOW)]
        if edge == "high" and have_ext["high"]:
            allowed += [lr for lr in by_lr if np.isclose(lr, EXT_HIGH)]
        pool = {lr: by_lr[lr] for lr in allowed} if stage == "final" else base_only
        best, second, items = pick_best_lr(pool, thresh)
        edge_after = None
        if stage == "final" and best is not None:
            lo = min(pool); hi = max(pool)
            if np.isclose(best, lo) and edge == "low" and have_ext["low"]:
                edge_after = "low"
            if np.isclose(best, hi) and edge == "high" and have_ext["high"]:
                edge_after = "high"

        decision[int(rank)] = dict(
            best_lr=float(best) if best is not None else None,
            second_best_lr=float(second) if second is not None else None,
            threshold=thresh,
            edge_of_base_grid=edge,
            extended=bool(edge and have_ext[edge] and stage == "final"),
            still_on_edge_after_extension=edge_after,
            table=[dict(lr=float(lr), mean_f1=float(f), n_seeds=int(nseed.get(lr, 0)),
                        crash_rate=float(crash.get(lr, 0.0))) for lr, f in items],
        )
    return decision, needs_ext


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV tune (run_pilot.py --eval_file calibration.csv => cột macro_f1 = F1 calibration)")
    ap.add_argument("--stage", choices=["base", "final", "maps"], default="final",
                    help="maps: đọc một tuning_decision.json có sẵn (vd của XLM-R) rồi chỉ xuất LR map phẳng")
    ap.add_argument("--f1_col", default="macro_f1")
    ap.add_argument("--thresh", type=float, default=LR_THRESH)
    ap.add_argument("--model", default="vinai/phobert-base")
    ap.add_argument("--data_dir", default="<DATA_DIR_PHOBERT>")
    ap.add_argument("--out", default=None)
    ap.add_argument("--flat_out", default=None, help="ghi {rank: lr} để dùng cho run_pilot.py --lr_map")
    ap.add_argument("--second_out", default=None, help="ghi {rank: LR tốt nhì} cho nhánh độ bền")
    a = ap.parse_args()

    if a.stage == "maps":
        dec = {int(k): v for k, v in json.load(open(a.csv, encoding="utf-8-sig")).items()}
        _write_maps(dec, a)
        return
    df = read_csv(a.csv)
    seeds = sorted(df["seed"].unique())
    if set(seeds) - set(TUNE_SEEDS):
        print(f"CẢNH BÁO: seed trong file tune {seeds} khác {TUNE_SEEDS}; seed tune phải tách khỏi seed chạy chính.")
    dec, needs = decide(df, a.f1_col, a.thresh, a.stage)

    print(f"\n=== Quy tắc LR (stage={a.stage}) | {len(df)} lượt | seed={seeds} ===")
    for r, d in sorted(dec.items()):
        cells = "  ".join(f"{t['lr']:g}={t['mean_f1']:.4f}(n={t['n_seeds']},sập {t['crash_rate']:.0%})" for t in d["table"])
        print(f"rank {r:>2}: best={d['best_lr']:g} second={d['second_best_lr']:g} biên={d['edge_of_base_grid']} | {cells}")
        if d["still_on_edge_after_extension"]:
            print(f"        LƯU Ý: sau mở rộng một bước LR vẫn nằm ở biên ({d['still_on_edge_after_extension']}); "
                  f"theo quy tắc KHÔNG mở rộng thêm, ghi vào hạn chế.")
    bad = [(r, t["lr"], t["n_seeds"]) for r, d in dec.items() for t in d["table"] if t["n_seeds"] < len(TUNE_SEEDS)]
    if bad:
        print("CẢNH BÁO thiếu seed:", bad)

    if a.stage == "base":
        if not needs:
            print("\nKhông rank nào cần mở rộng lưới. Có thể chạy `--stage final` ngay trên cùng CSV.")
        for side, lr in (("low", EXT_LOW), ("high", EXT_HIGH)):
            rk = [r for r, s in sorted(needs.items()) if s == side]
            if rk:
                print(f"\nCần mở rộng biên {side} (LR {lr:g}) cho rank {rk}. Lệnh:")
                print(f"python code/run_pilot.py --data_dir {a.data_dir} --model {a.model} "
                      f"--out results/ext/phobert/tune_ext_{side}.csv --cell TUNE_PHO --sweep_global "
                      f"--ranks {' '.join(map(str, rk))} --lr_list {lr:g} --epochs_list 6 "
                      f"--seeds 400 401 402 --precision fp16 --eval_file calibration.csv")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(dec, open(a.out, "w", encoding="utf-8"), indent=1)
        print(f"\nĐã lưu {a.out}")
    if a.stage == "final":
        _write_maps(dec, a)


def _write_maps(dec, a):
    miss = [r for r in (1, 4, 8, 16, 32) if r not in dec or dec[r]["best_lr"] is None]
    if miss:
        raise SystemExit(f"LR map thiếu rank {miss}: run_pilot.py sẽ BỎ QUA im lặng các rank này!")
    for path, key in ((a.flat_out, "best_lr"), (a.second_out, "second_best_lr")):
        if not path:
            continue
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        json.dump({str(r): d[key] for r, d in sorted(dec.items())}, open(path, "w"), indent=1)
        print(f"Đã lưu LR map phẳng ({key}) -> {path}")


if __name__ == "__main__":
    main()

# code/ext_lr_control_table.py - Same-session LR control (PhoBERT, rank 8/16/32, seeds 400-402).
# Compares LR 1e-4 (tune_ctrl_1e4.csv) with 5e-5 (tuning_5e5.csv) on CALIBRATION macro-F1 only,
# applies the pre-registered tie threshold (0.005 -> smaller LR) and writes a JSON table.
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESH = 0.005


def main():
    c = pd.read_csv(os.path.join(ROOT, "results/ext/phobert/tune_ctrl_1e4.csv"))
    t = pd.read_csv(os.path.join(ROOT, "results/tuning/tuning_5e5.csv"))
    assert (c["eval_file"] == "calibration.csv").all(), "control runs must be scored on calibration"
    t = t[t["lr"] == 5e-5]  # tuning_5e5.csv was scored on test.csv: use cal_macro_f1 only
    out = {}
    print("rank | 1e-4 cal F1 | 5e-5 cal F1 | diff | paired SD | choice")
    for r in sorted(c["rank"].unique()):
        a = c[c["rank"] == r].set_index("seed")["cal_macro_f1"]
        b = t[t["rank"] == r].set_index("seed")["cal_macro_f1"]
        d = (a - b.reindex(a.index))
        diff = float(a.mean() - b.mean())
        choice = 1e-4 if diff > THRESH else 5e-5
        out[int(r)] = {"mean_1e-4": float(a.mean()), "mean_5e-5": float(b.mean()), "diff": diff,
                       "paired_sd": float(d.std(ddof=1)), "n": int(len(a)), "choice": choice,
                       "margin_over_threshold": diff - THRESH}
        print(f"{r:>4} | {a.mean():.4f} | {b.mean():.4f} | {diff:+.4f} | {d.std(ddof=1):.4f} | {choice:g}")
    p = os.path.join(ROOT, "results/ext/phobert/lr_control_table.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    print("Written results/ext/phobert/lr_control_table.json")


if __name__ == "__main__":
    main()

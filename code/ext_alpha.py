"""ext_alpha.py - độ nhạy hệ số α (α = 2r so với α cố định = 16) cho CẢ HAI mô hình.

Gợi ý hiện có (3 seed, chỉ XLM-R): phần NLL tăng theo rank liên quan đến α = 2r. EXT-1 kiểm lại với 8 seed,
rank {1, 8, 32}, LR cố định 2e-4, cho cả PhoBERT và XLM-R. Mỗi seed -> tổng hiệu ứng rank (1->32) cho NLL và F1
dưới từng chế độ α; so sánh ghép cặp theo seed: (fixed:16) - (2r). THĂM DÒ, không thuộc họ Holm.

  python code/ext_alpha.py --csv_p results/ext/phobert/B1X_merged.csv results/ext/phobert/ALPHA_P_gpu*.csv \
                           --csv_x results/posthoc/V1b_merged.csv results/ext/xlmr/V1b_ext_gpu*.csv results/ext/xlmr/ALPHA_X_gpu*.csv
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_common import read_csv, per_seed_effects, one_sample  # noqa: E402
from scipy import stats  # noqa: E402


def load(patterns):
    paths = []
    for p in patterns:
        paths += sorted(glob.glob(p)) or ([p] if os.path.exists(p) else [])
    return pd.concat([read_csv(p) for p in paths]).drop_duplicates("run_id")


def analyse(df, ranks, lr):
    d = df[(df["epochs"] == 6) & np.isclose(df["lr"], lr) & df["rank"].isin(ranks)]
    out = {}
    per = {}
    for mode, g in d.groupby("alpha_mode"):
        f, _ = per_seed_effects(g, "macro_f1", ranks)
        n, _ = per_seed_effects(g, "nll", ranks)
        per[mode] = f.set_index("seed")[["total"]].rename(columns={"total": "f1"}).join(
            n.set_index("seed")[["total"]].rename(columns={"total": "nll"}))
        out[mode] = dict(n_seeds=len(per[mode]), f1_total=float(per[mode]["f1"].mean()), nll_total=float(per[mode]["nll"].mean()))
    if "2r" in per and "fixed:16" in per:
        j = per["fixed:16"].join(per["2r"], lsuffix="_fix", rsuffix="_2r", how="inner")
        for y in ("nll", "f1"):
            diff = (j[f"{y}_fix"] - j[f"{y}_2r"]).to_numpy()
            if len(diff) >= 3:
                m, se, df_ = one_sample(diff)
                q = stats.t.ppf(0.975, df_)
                p = float(2 * (1 - stats.t.cdf(abs(m / se), df_))) if se > 0 else (1.0 if m == 0 else 0.0)
                out[f"paired_{y}_fixed16_minus_2r"] = dict(est=m, ci95=[m - q * se, m + q * se], n=len(diff), p=p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv_p", nargs="+"); ap.add_argument("--csv_x", nargs="+")
    ap.add_argument("--ranks", type=int, nargs="*", default=[1, 8, 32])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {}
    for tag, name in (("csv_p", "PhoBERT"), ("csv_x", "XLM-R")):
        if getattr(a, tag):
            res[name] = analyse(load(getattr(a, tag)), a.ranks, a.lr)
            print(f"\n=== {name} (rank {a.ranks}, LR {a.lr:g}, 6 epoch) ===")
            for k, v in res[name].items():
                if k.startswith("paired"):
                    print(f"{k}: {v['est']:+.4f} CI95 [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] n={v['n']} p={v['p']:.3g}")
                else:
                    print(f"alpha={k}: {v['n_seeds']} seed, tổng hiệu ứng rank: F1 {v['f1_total']:+.4f}, NLL {v['nll_total']:+.4f}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, default=float)


if __name__ == "__main__":
    main()

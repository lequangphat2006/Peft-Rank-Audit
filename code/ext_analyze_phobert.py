"""ext_analyze_phobert.py - phân tích mô tả + độ bền cho PhoBERT (5 rank x 16 seed x 6 epoch).

Không nằm trong họ xác nhận (họ xác nhận E1-E3 ở code/ext_confirmatory.py). Script này tạo:
  A. Bảng theo rank: Macro-F1, SD, F1 lớp neutral, NLL, số seed, tỉ lệ lượt sập.
  B. Ước lượng kiểu H1/H2 cho riêng PhoBERT: tổng hiệu ứng rank 1->32, TOST delta 0,01 và 0,02, NLL.
  C. Kiểm lệch tuyến tính (log2(rank) tuyến tính so với rank dạng phân loại) và "rank bão hòa".
  D. Độ nhạy theo số seed: 8 seed đầu (như B1 cũ) so với đủ 16 seed.
  E. Nhánh độ bền: LR theo rank (chính) | LR cố định 2e-4 | LR tốt nhì | bản đồ LR cũ của V4.

Ví dụ:
  python code/ext_analyze_phobert.py results/ext/phobert/B1_ext_merged.csv \
      --lr_map results/ext/phobert/phobert_lr_map_v2.json \
      --decision results/ext/phobert/tuning_decision.json \
      --legacy_lr_map results/v4_phobert/phobert_lr_map.json \
      --out results/ext/phobert/analysis_phobert.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_common import (read_csv, load_lr_map, load_second_best, select_branch, select_fixed_lr,  # noqa: E402
                        per_seed_effects, one_sample, tost_from, onesided_pos_p, mark_crash)


def table_by_rank(d):
    d = mark_crash(d)
    g = d.groupby("rank")
    t = pd.DataFrame({
        "n_seeds": g["seed"].nunique(),
        "lr": g["lr"].agg(lambda s: ",".join(f"{x:g}" for x in sorted(s.unique()))),
        "f1_mean": g["macro_f1"].mean(), "f1_sd": g["macro_f1"].std(ddof=1),
        "f1_neu": g["f1_neu"].mean() if "f1_neu" in d.columns else np.nan,
        "nll_mean": g["nll"].mean(), "nll_sd": g["nll"].std(ddof=1),
        "crash_rate": g["crash"].mean(),
    })
    return t.round(4)


def estimate(d, ranks, deltas=(0.01, 0.02)):
    """Tổng hiệu ứng rank (độ dốc x khoảng log2) cho F1 và NLL, TOST và kiểm định NLL tăng."""
    out = {}
    f, drop = per_seed_effects(d, "macro_f1", ranks)
    n, _ = per_seed_effects(d, "nll", ranks)
    out["n_seeds_used"], out["seeds_dropped"] = int(len(f)), drop
    if len(f) < 3:
        return out
    m, se, df = one_sample(f["total"])
    out["f1_total"] = dict(est=m, se=se, df=df, ci95=[m - 1.0 * se * _t(df, .975), m + se * _t(df, .975)])
    for dl in deltas:
        p, ci90 = tost_from(m, se, df, dl)
        out[f"tost_delta_{dl:g}"] = dict(p=p, ci90=ci90, equivalent=bool(p < 0.05))
    md, sed, dfd = one_sample(f["diff"])
    p, ci90 = tost_from(md, sed, dfd, 0.01)
    out["f1_diff_hi_lo_direct"] = dict(est=md, se=sed, ci90=ci90, tost_p_delta_0_01=p)
    mn, sen, dfn = one_sample(n["total"])
    out["nll_total"] = dict(est=mn, se=sen, df=dfn, p_increase=onesided_pos_p(mn, sen, dfn),
                            ci95=[mn - sen * _t(dfn, .975), mn + sen * _t(dfn, .975)])
    return out


def _t(df, q):
    from scipy import stats
    return float(stats.t.ppf(q, df))


def nonlinearity(d, ranks):
    d = d[d["rank"].isin(ranks)].copy()
    d["lr2"] = np.log2(d["rank"])
    d["rk"] = d["rank"].astype(str)
    red = smf.ols("macro_f1 ~ C(seed) + lr2", d).fit()
    full = smf.ols("macro_f1 ~ C(seed) + C(rk)", d).fit()
    a = sm.stats.anova_lm(red, full)
    return dict(F=float(a["F"].iloc[1]), df_num=int(a["df_diff"].iloc[1]), p=float(a["Pr(>F)"].iloc[1]))


def saturation(t, tol=0.005):
    """Rank nhỏ nhất mà F1 trung bình nằm trong `tol` của F1 trung bình lớn nhất (mô tả, không phải kiểm định)."""
    best = t["f1_mean"].max()
    for r, row in t.iterrows():
        if best - row["f1_mean"] <= tol:
            return int(r)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--ranks", type=int, nargs="*", default=[1, 4, 8, 16, 32])
    ap.add_argument("--lr_map", required=True, help="LR map v2 (nhánh chính)")
    ap.add_argument("--decision", default=None, help="tuning_decision.json (để lấy LR tốt nhì)")
    ap.add_argument("--legacy_lr_map", default=None, help="phobert_lr_map.json của V4 (kiểm tra độ nhạy)")
    ap.add_argument("--fixed_lr", type=float, default=2e-4)
    ap.add_argument("--first_seeds", type=int, default=8, help="số seed đầu cho độ nhạy 8 vs 16")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    df = read_csv(a.csv)
    df = df[(df["epochs"] == a.epochs) & df["rank"].isin(a.ranks)]
    lr_map = load_lr_map(a.lr_map)
    main_d = select_branch(df, lr_map)
    out = dict(csv=a.csv, ranks=a.ranks, epochs=a.epochs, lr_map=lr_map)

    print(f"\n=== A. Nhánh chính (LR theo rank) | {main_d['seed'].nunique()} seed ===")
    tab = table_by_rank(main_d)
    print(tab.to_string())
    out["table_main"] = tab.reset_index().to_dict(orient="records")

    print("\n=== B. Ước lượng cho PhoBERT (kiểu H1/H2) ===")
    est = estimate(main_d, a.ranks)
    out["estimate_main"] = est
    if "f1_total" in est:
        f = est["f1_total"]
        print(f"F1 tổng rank {min(a.ranks)}->{max(a.ranks)}: {f['est']:+.4f} CI95 [{f['ci95'][0]:+.4f},{f['ci95'][1]:+.4f}]")
        for dl in (0.01, 0.02):
            t = est[f"tost_delta_{dl:g}"]
            print(f"  TOST delta={dl:g}: p={t['p']:.4f} CI90 [{t['ci90'][0]:+.4f},{t['ci90'][1]:+.4f}] -> "
                  f"{'tương đương' if t['equivalent'] else 'KHÔNG chứng minh tương đương'}")
        n = est["nll_total"]
        print(f"NLL tổng: {n['est']:+.4f} CI95 [{n['ci95'][0]:+.4f},{n['ci95'][1]:+.4f}] p(tăng)={n['p_increase']:.4f}")

    print("\n=== C. Lệch tuyến tính và bão hòa ===")
    if len(a.ranks) >= 3 and main_d["seed"].nunique() >= 3:
        nl = nonlinearity(main_d, a.ranks)
        sat = saturation(tab)
        out["nonlinearity"], out["saturation_rank_tol_0.005"] = nl, sat
        print(f"F-test log2(rank) tuyến tính vs phân loại: F={nl['F']:.3f} (df {nl['df_num']}) p={nl['p']:.4f}")
        print(f"Rank nhỏ nhất đạt trong 0,005 của F1 lớn nhất: {sat}")

    print(f"\n=== D. Độ nhạy theo số seed ({a.first_seeds} vs tất cả) ===")
    seeds = sorted(main_d["seed"].unique())
    if len(seeds) > a.first_seeds:
        sub = main_d[main_d["seed"].isin(seeds[:a.first_seeds])]
        fresh = main_d[main_d["seed"].isin(seeds[a.first_seeds:])]   # seed mới (308-315): chưa từng xuất hiện trong B1 cũ
        for tag, d in ((f"{a.first_seeds} seed đầu", sub), (f"{len(seeds) - a.first_seeds} seed mới", fresh),
                       (f"{len(seeds)} seed", main_d)):
            e = estimate(d, a.ranks)
            if "f1_total" in e:
                f = e["f1_total"]
                print(f"{tag}: F1 tổng {f['est']:+.4f} CI95 [{f['ci95'][0]:+.4f},{f['ci95'][1]:+.4f}] "
                      f"| TOST 0,01 p={e['tost_delta_0.01']['p']:.4f}")
                out[f"seed_sensitivity_{tag}"] = e
    else:
        print("Chưa đủ seed để so sánh.")

    print("\n=== E. Các nhánh độ bền ===")
    branches = {"fixed_lr": select_fixed_lr(df, a.fixed_lr)}
    if a.decision and os.path.exists(a.decision):
        sb = load_second_best(a.decision)
        branches["second_best_lr"] = select_branch(df, sb)
    if a.legacy_lr_map and os.path.exists(a.legacy_lr_map):
        leg = load_lr_map(a.legacy_lr_map)
        branches["legacy_v4_lr_map"] = select_branch(df, leg)
    for name, d in branches.items():
        have = sorted(int(x) for x in d["rank"].unique())
        if len(have) < 3 or d["seed"].nunique() < 3:
            print(f"{name}: chưa đủ dữ liệu (rank có: {have}, seed: {d['seed'].nunique()})")
            continue
        e = estimate(d, have)
        out[f"branch_{name}"] = dict(ranks=have, **e)
        f = e["f1_total"]
        print(f"{name}: rank {have}, {e['n_seeds_used']} seed, F1 tổng {f['est']:+.4f} "
              f"CI95 [{f['ci95'][0]:+.4f},{f['ci95'][1]:+.4f}]")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
        print(f"\nĐã lưu {a.out}")


if __name__ == "__main__":
    main()

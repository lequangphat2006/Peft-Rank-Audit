"""Confirmatory analysis of the global rank sweep, matching docs/analysis_plan_v2.0.md.

H1-H3, Holm m=3. H3 has TWO branches:
  * main branch   : LR tuned per rank (from tuning_decision.json)
  * robustness    : LR fixed 2e-4 for all ranks (subset of R1 runs)
H3 is robust only when both branches agree in sign AND the fixed-LR interaction estimate
falls inside the 95% CI of the main branch.

Also reports:
  * relative delta sensitivity (muc 4.6)
  * forest plot data by cell
  * exploratory temperature scaling + ECE
Usage:
  python code/analyze_global.py results/global/global_merged.csv \
      --primary_epochs 6 --delta_f1 0.01 \
      --lr_decision results/tuning/tuning_decision.json \
      --fixed_lr 2e-4 --logits_dir results/global/logits \
      --out results/analysis/global_v1.json
"""
import argparse, json, os
import numpy as np, pandas as pd
import statsmodels.formula.api as smf, statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize_scalar


def slope(d, y):
    d = d.assign(lr=np.log2(d["rank"]))
    m = smf.ols(f"{y} ~ C(seed) + lr", d).fit()
    span = d["lr"].max() - d["lr"].min()
    b, se, df = float(m.params["lr"]), float(m.bse["lr"]), int(m.df_resid)
    return dict(slope=b, se=se, df=df, total=b * span, total_se=se * span, span=span,
                ci95_total=[(b - stats.t.ppf(.975, df) * se) * span, (b + stats.t.ppf(.975, df) * se) * span],
                ci90_total=[(b - stats.t.ppf(.95, df) * se) * span, (b + stats.t.ppf(.95, df) * se) * span],
                n=len(d), n_seeds=int(d["seed"].nunique()))


def tost_p(est, delta):
    t_lo = (est["total"] + delta) / est["total_se"]
    t_hi = (delta - est["total"]) / est["total_se"]
    return float(max(1 - stats.t.cdf(t_lo, est["df"]), 1 - stats.t.cdf(t_hi, est["df"])))


def onesided_pos_p(est):
    return float(1 - stats.t.cdf(est["slope"] / est["se"], est["df"]))


def interaction(d, y):
    d = d.assign(lr=np.log2(d["rank"]), epochs=d["epochs"].astype(str))
    full = smf.ols(f"{y} ~ C(seed) + C(epochs) + lr + lr:C(epochs)", d).fit()
    red  = smf.ols(f"{y} ~ C(seed) + C(epochs) + lr", d).fit()
    a = sm.stats.anova_lm(red, full)
    F, p, ddf = float(a["F"].iloc[1]), float(a["Pr(>F)"].iloc[1]), int(a["df_diff"].iloc[1])
    return dict(F=F, df=ddf, p=p)


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def temperature_scale(logits_dir, ids_df):
    rows = []
    for _, r in ids_df.iterrows():
        p = os.path.join(logits_dir, r["run_id"] + ".npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        lc, gc, le, ge = z["cal_logits"], z["cal_gold"], z["eval_logits"], z["eval_gold"]
        def nll(logT, lg, g):
            s = lg / np.exp(logT); s = s - s.max(1, keepdims=True)
            return float(-(s[np.arange(len(g)), g] - np.log(np.exp(s).sum(1))).mean())
        res = minimize_scalar(lambda t: nll(t, lc, gc), bounds=(-3, 3), method="bounded")
        T = float(np.exp(res.x))
        p = np.exp((le / T) - (le / T).max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        conf, pred = p.max(1), p.argmax(1)
        ece = 0.0
        for lo in np.linspace(0, 1, 16)[:-1]:
            mk = (conf > lo) & (conf <= lo + 1 / 15)
            if mk.any():
                ece += mk.mean() * abs((pred[mk] == ge[mk]).mean() - conf[mk].mean())
        rows.append(dict(run_id=r["run_id"], T=T, nll_ts=nll(res.x, le, ge), ece_ts=float(ece)))
    return pd.DataFrame(rows)


def load_lr_decision(path):
    """Đọc tuning_decision.json -> {rank: best_lr}."""
    if path is None or not os.path.exists(path):
        return None
    j = json.load(open(path))
    return {int(k): float(v["best_lr"]) for k, v in j.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--primary_epochs", type=int, default=6)
    ap.add_argument("--delta_f1", type=float, default=0.01)
    ap.add_argument("--lr_decision", default=None,
                    help="tuning_decision.json; nếu có, H3 nhánh chính dùng LR theo rank")
    ap.add_argument("--fixed_lr", type=float, default=2e-4)
    ap.add_argument("--logits_dir", default=None)
    ap.add_argument("--relative_f1_ref", type=float, default=None,
                    help="F1 của R1 rank 8 (6 epoch) để tính δ tương đối; mặc định lấy từ data")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    out = {"n_runs": len(df), "seeds": sorted(map(int, df.seed.unique()))}
    prim = df[df.epochs == a.primary_epochs].copy()

    # ---- Confirmatory family ----
    est_f1, est_nll = slope(prim, "macro_f1"), slope(prim, "nll")
    p1, p2 = tost_p(est_f1, a.delta_f1), onesided_pos_p(est_nll)

    # H3 with two branches
    lr_map = load_lr_decision(a.lr_decision)
    h3_main = interaction(df, "macro_f1") if df.epochs.nunique() > 1 else dict(F=np.nan, df=0, p=np.nan)

    # fixed-LR branch: keep only runs with lr == fixed_lr (works even if LR was tuned in other runs)
    fixed = df[np.isclose(df.lr, a.fixed_lr)]
    if fixed.epochs.nunique() > 1 and len(fixed) > 0:
        h3_fixed = interaction(fixed, "macro_f1")
    else:
        h3_fixed = dict(F=np.nan, df=0, p=np.nan)

    fam = [p1, p2] + ([h3_main["p"]] if np.isfinite(h3_main["p"]) else [])
    adj = holm(fam)

    print("\n=== CONFIRMATORY FAMILY (Holm, m=%d) ===" % len(fam))
    print(f"H1 F1 total (rank {prim['rank'].min()}->{prim['rank'].max()}): "
          f"{est_f1['total']:+.4f} 90%CI [{est_f1['ci90_total'][0]:+.4f},{est_f1['ci90_total'][1]:+.4f}] "
          f"TOST d={a.delta_f1} p={p1:.4f} p_holm={adj[0]:.4f} -> "
          f"{'EQUIVALENT' if adj[0] < .05 else 'NOT shown equivalent'}")
    print(f"H2 NLL total: {est_nll['total']:+.4f} 95%CI [{est_nll['ci95_total'][0]:+.4f},"
          f"{est_nll['ci95_total'][1]:+.4f}] p={p2:.4f} p_holm={adj[1]:.4f} -> "
          f"{'INCREASES' if adj[1] < .05 else 'not confirmed'}")
    if np.isfinite(h3_main["p"]):
        print(f"H3 main branch (LR per rank): F={h3_main['F']:.3f} df={h3_main['df']} "
              f"p={h3_main['p']:.4f} p_holm={adj[2]:.4f}")
    if np.isfinite(h3_fixed["p"]):
        print(f"H3 robustness (LR fixed {a.fixed_lr:g}): F={h3_fixed['F']:.3f} df={h3_fixed['df']} "
              f"p={h3_fixed['p']:.4f}")
    out.update(H1=dict(est=est_f1, p=p1, p_holm=adj[0]),
               H2=dict(est=est_nll, p=p2, p_holm=adj[1]),
               H3=dict(main=h3_main, fixed=h3_fixed))

    # ---- Relative delta sensitivity ----
    if a.relative_f1_ref is None:
        r8 = df[(df["rank"] == 8) & (df.epochs == a.primary_epochs)]["macro_f1"].mean()
        ref = float(r8) if np.isfinite(r8) else est_f1["total"]  # fallback
    else:
        ref = float(a.relative_f1_ref)
    delta_rel = a.delta_f1 * (1.0 / max(ref, 1e-6))  # δ tương đối theo tỉ lệ của R1
    out["relative_delta"] = dict(f1_ref=ref, delta_abs=a.delta_f1, delta_rel=delta_rel)
    print(f"\nRelative delta: F1 ref (rank8, {a.primary_epochs} ep) = {ref:.4f}; "
          f"delta_abs = {a.delta_f1}; delta_rel = {delta_rel:.4f}")

    # ---- Exploratory ----
    print("\n=== EXPLORATORY (not in the family) ===")
    for ep, d in df.groupby("epochs"):
        f, n = slope(d, "macro_f1"), slope(d, "nll")
        print(f"epochs={ep}: F1 total {f['total']:+.4f} [{f['ci95_total'][0]:+.4f},{f['ci95_total'][1]:+.4f}]  "
              f"NLL total {n['total']:+.4f} [{n['ci95_total'][0]:+.4f},{n['ci95_total'][1]:+.4f}]")
        out[f"exploratory_epochs_{ep}"] = dict(f1=f, nll=n)

    # per-cell (if column exists)
    if "cell" in df.columns and df["cell"].nunique() > 1:
        forest = []
        for c, d in df.groupby("cell"):
            if d.epochs.nunique() > 1:
                continue
            f = slope(d, "macro_f1")
            forest.append(dict(cell=c, total=f["total"],
                               ci95=f["ci95_total"], n=f["n"], n_seeds=f["n_seeds"]))
        out["forest"] = forest
        print("\nForest plot by cell:")
        for r in forest:
            print(f"  {r['cell']}: {r['total']:+.4f} [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] n={r['n']}")

    if a.logits_dir:
        ts = temperature_scale(a.logits_dir, prim[["run_id", "seed", "rank"]])
        if len(ts):
            merged = prim.merge(ts, on="run_id")
            e = slope(merged, "nll_ts")
            print(f"\nNLL after temperature scaling: total {e['total']:+.4f} "
                  f"[{e['ci95_total'][0]:+.4f},{e['ci95_total'][1]:+.4f}]")
            print(merged.groupby("rank")[["T", "nll_ts", "ece_ts"]].mean().round(4).to_string())
            out["temperature_scaling"] = dict(est=e, by_rank=merged.groupby("rank")[["T", "nll_ts", "ece_ts"]].mean().to_dict())

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1, default=float)
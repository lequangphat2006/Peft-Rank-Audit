"""Confirmatory analysis of the global rank sweep, exactly as locked in docs/analysis_plan.md.

Input : merged CSV of run_pilot.py v2 (columns seed, rank, epochs, macro_f1, nll, [cal_nll], run_id)
Models: metric ~ C(seed) + log2(rank)            (seed = fixed block, one run per seed x rank x budget)
        H3    : metric ~ C(seed) + C(epochs) + lr + lr:C(epochs)  (same seeds in every budget)
Family: H1, H2, H3 with Holm (m = 3). Everything else printed here is EXPLORATORY and labelled so.
Usage : python code/analyze_global.py results/global/global_merged.csv --primary_epochs 6 --delta_f1 0.005 \
            --logits_dir results/global/logits --out results/analysis/global_v1.json
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
    t_lo = (est["total"] + delta) / est["total_se"]; t_hi = (delta - est["total"]) / est["total_se"]
    return float(max(1 - stats.t.cdf(t_lo, est["df"]), 1 - stats.t.cdf(t_hi, est["df"])))


def onesided_pos_p(est):
    return float(1 - stats.t.cdf(est["slope"] / est["se"], est["df"]))


def interaction_p(d, y):
    d = d.assign(lr=np.log2(d["rank"]), epochs=d["epochs"].astype(str))
    full = smf.ols(f"{y} ~ C(seed) + C(epochs) + lr + lr:C(epochs)", d).fit()
    red = smf.ols(f"{y} ~ C(seed) + C(epochs) + lr", d).fit()
    a = sm.stats.anova_lm(red, full)
    F, p, ddf = float(a["F"].iloc[1]), float(a["Pr(>F)"].iloc[1]), int(a["df_diff"].iloc[1])
    assert np.isfinite(p), "interaction test failed -> do not report n.s."
    return dict(F=F, df=ddf, p=p)


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def temperature_scale(logits_dir, ids_df):
    """EXPLORATORY: per-run temperature fitted on calibration logits, applied to the evaluation logits."""
    rows = []
    for _, r in ids_df.iterrows():
        z = np.load(os.path.join(logits_dir, r["run_id"] + ".npz"))
        lc, gc, le, ge = z["cal_logits"], z["cal_gold"], z["eval_logits"], z["eval_gold"]

        def nll(logT, lg, g):
            s = lg / np.exp(logT); s = s - s.max(1, keepdims=True)
            return float(-(s[np.arange(len(g)), g] - np.log(np.exp(s).sum(1))).mean())
        res = minimize_scalar(lambda t: nll(t, lc, gc), bounds=(-3, 3), method="bounded")
        p = np.exp((le / np.exp(res.x)) - (le / np.exp(res.x)).max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        conf, pred = p.max(1), p.argmax(1); ece = 0.0
        for lo in np.linspace(0, 1, 16)[:-1]:
            mk = (conf > lo) & (conf <= lo + 1 / 15)
            if mk.any(): ece += mk.mean() * abs((pred[mk] == ge[mk]).mean() - conf[mk].mean())
        rows.append(dict(run_id=r["run_id"], T=float(np.exp(res.x)), nll_ts=nll(res.x, le, ge), ece_ts=float(ece)))
    return ids_df.merge(pd.DataFrame(rows), on="run_id")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--primary_epochs", type=int, default=6)
    ap.add_argument("--delta_f1", type=float, default=0.005); ap.add_argument("--logits_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.csv); out = {"n_runs": len(df), "seeds": sorted(map(int, df.seed.unique()))}
    prim = df[df.epochs == a.primary_epochs]
    print(f"Primary budget: {a.primary_epochs} epochs, seeds={out['seeds']}")
    est_f1, est_nll = slope(prim, "macro_f1"), slope(prim, "nll")
    p1, p2 = tost_p(est_f1, a.delta_f1), onesided_pos_p(est_nll)
    h3 = interaction_p(df, "macro_f1") if df.epochs.nunique() > 1 else dict(F=np.nan, df=0, p=np.nan)
    fam = [p1, p2, h3["p"]] if np.isfinite(h3["p"]) else [p1, p2]
    adj = holm(fam)
    print("\n=== CONFIRMATORY FAMILY (Holm, m=%d) ===" % len(fam))
    print(f"H1 F1 total effect rank {prim['rank'].min()}->{prim['rank'].max()}: {est_f1['total']:+.4f} "
          f"90%CI [{est_f1['ci90_total'][0]:+.4f}, {est_f1['ci90_total'][1]:+.4f}]  TOST delta={a.delta_f1}  "
          f"p={p1:.4f} p_holm={adj[0]:.4f} -> {'EQUIVALENT' if adj[0] < .05 else 'NOT shown equivalent'}")
    print(f"H2 NLL total effect: {est_nll['total']:+.4f} 95%CI [{est_nll['ci95_total'][0]:+.4f}, "
          f"{est_nll['ci95_total'][1]:+.4f}]  one-sided p={p2:.4f} p_holm={adj[1]:.4f} -> "
          f"{'NLL INCREASES with rank' if adj[1] < .05 else 'not confirmed'}")
    if len(fam) == 3:
        print(f"H3 rank x epochs (F1): F={h3['F']:.3f} df={h3['df']} p={h3['p']:.4f} p_holm={adj[2]:.4f}")
    out.update(H1=dict(est=est_f1, p=p1, p_holm=adj[0]), H2=dict(est=est_nll, p=p2, p_holm=adj[1]), H3=h3)
    print("\n=== EXPLORATORY (not in the family) ===")
    for ep, d in df.groupby("epochs"):
        f, n = slope(d, "macro_f1"), slope(d, "nll")
        print(f"epochs={ep}: F1 total {f['total']:+.4f} [{f['ci95_total'][0]:+.4f},{f['ci95_total'][1]:+.4f}]  "
              f"NLL total {n['total']:+.4f} [{n['ci95_total'][0]:+.4f},{n['ci95_total'][1]:+.4f}]")
        out[f"exploratory_epochs_{ep}"] = dict(f1=f, nll=n)
    if a.logits_dir:
        ts = temperature_scale(a.logits_dir, prim[["run_id", "seed", "rank"]])
        e = slope(ts, "nll_ts"); print(f"NLL after temperature scaling (calibration split): total {e['total']:+.4f} "
                                       f"[{e['ci95_total'][0]:+.4f},{e['ci95_total'][1]:+.4f}]; mean T by rank:")
        print(ts.groupby("rank")[["T", "nll_ts", "ece_ts"]].mean().round(4).to_string()); out["temperature_scaling"] = e
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1, default=float)

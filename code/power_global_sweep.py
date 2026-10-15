"""How many seeds does the global rank sweep need?  (run BEFORE launching Phase 1; recompute after 8 seeds)

Reads a per-run CSV (columns seed, rank, macro_f1, nll; optional module for the Stage 1c layout), estimates the
residual SD of a run after removing seed (and module x rank) effects, then, for a design with ranks R x N seeds
(one run per rank per seed), reports
  * SE of the pooled slope per log2(rank) and of the total effect over log2(max/min rank) doublings,
  * the half-width of the 90% CI (TOST) for the total effect and whether it is <= delta,
  * the power of a one-sided test that the NLL slope is > 0 for a given true total effect.
Assumption (state it in the paper): the residual SD of a GLOBAL-rank run is similar to that of a single-module run.
Usage: python power_global_sweep.py results/pilot/stage1c_merged.csv --delta_f1 0.005 --nll_effect 0.007
"""
import argparse, json
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def resid_sd(df, y):
    f = f"{y} ~ C(seed) + C(module)*C(rank)" if "module" in df else f"{y} ~ C(seed) + C(rank)"
    m = smf.ols(f, df).fit()
    return float(np.sqrt(m.scale)), int(m.df_resid)


def design(ranks, n):
    x = np.log2(np.asarray(ranks, float)); sxx = ((x - x.mean()) ** 2).sum() * n
    return x, sxx


def table(sd, ranks, delta, nll_effect, ns, alpha=0.05):
    x, _ = design(ranks, 1); span = x.max() - x.min(); rows = []
    for n in ns:
        _, sxx = design(ranks, n); df = n * len(ranks) - n - 1
        se_slope = sd / np.sqrt(sxx); se_total = se_slope * span
        hw90 = stats.t.ppf(0.95, df) * se_total
        ncp = nll_effect / se_total; tc = stats.t.ppf(1 - alpha, df)
        power = float(1 - stats.nct.cdf(tc, df, ncp))
        rows.append(dict(n_seeds=n, runs=n * len(ranks), se_total=round(se_total, 5), ci90_halfwidth=round(hw90, 5),
                         tost_ok=bool(hw90 <= delta), power_nll_slope=round(power, 3)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--ranks", type=int, nargs="*", default=[1, 4, 16, 32])
    ap.add_argument("--delta_f1", type=float, default=0.005); ap.add_argument("--nll_effect", type=float, default=0.007)
    ap.add_argument("--ns", type=int, nargs="*", default=[4, 6, 8, 10, 12, 16])
    a = ap.parse_args()
    df = pd.read_csv(a.csv); out = {}
    for y, tgt in (("macro_f1", a.delta_f1), ("nll", a.delta_f1)):
        sd, dfr = resid_sd(df, y); print(f"\n{y}: residual SD = {sd:.5f} (df={dfr})")
        t = table(sd, a.ranks, a.delta_f1 if y == "macro_f1" else 0.0, a.nll_effect, a.ns)
        print(pd.DataFrame(t).drop(columns=["power_nll_slope"] if y == "macro_f1" else ["tost_ok"]).to_string(index=False))
        out[y] = dict(sd=sd, table=t)
    print(json.dumps({k: v["sd"] for k, v in out.items()}))

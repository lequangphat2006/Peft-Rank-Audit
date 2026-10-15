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


def tost_power(se_total, df, delta, true_effect, nsim=20000, seed=0):
    """P(TOST succeeds) = P(|est| + t_.95 * se_hat <= delta). 'CI half-width <= delta' is only NECESSARY."""
    rng = np.random.default_rng(seed)
    est = true_effect + se_total * rng.standard_normal(nsim)
    se_hat = se_total * np.sqrt(rng.chisquare(df, nsim) / df)
    return float(np.mean(np.abs(est) + stats.t.ppf(0.95, df) * se_hat <= delta))


def table(sd, ranks, delta, nll_effect, ns, alpha=0.05, f1_true=(0.0, -0.0013)):
    x, _ = design(ranks, 1); span = x.max() - x.min(); rows = []
    for n in ns:
        _, sxx = design(ranks, n); df = n * len(ranks) - n - 1
        se_slope = sd / np.sqrt(sxx); se_total = se_slope * span
        hw90 = stats.t.ppf(0.95, df) * se_total
        ncp = nll_effect / se_total; tc = stats.t.ppf(1 - alpha, df)
        power = float(1 - stats.nct.cdf(tc, df, ncp))
        row = dict(n_seeds=n, runs=n * len(ranks), se_total=round(se_total, 5), ci90_halfwidth=round(hw90, 5),
                   halfwidth_le_delta=bool(hw90 <= delta), power_nll_slope=round(power, 3))
        for mu in f1_true:
            row[f"tost_power_mu{mu:g}"] = round(tost_power(se_total, df, delta, mu), 3)
        rows.append(row)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--ranks", type=int, nargs="*", default=[1, 4, 16, 32])
    ap.add_argument("--delta_f1", type=float, nargs="*", default=[0.005, 0.0075, 0.01]); ap.add_argument("--nll_effect", type=float, default=0.007)
    ap.add_argument("--ns", type=int, nargs="*", default=[4, 6, 8, 10, 12, 16])
    a = ap.parse_args()
    df = pd.read_csv(a.csv); out = {}
    for y, dl in [("macro_f1", d) for d in a.delta_f1] + [("nll", 0.0)]:
        sd, dfr = resid_sd(df, y); print(f"\n{y}: residual SD = {sd:.5f} (df={dfr})" + (f"  delta={dl}" if y == "macro_f1" else ""))
        t = table(sd, a.ranks, dl, a.nll_effect, a.ns)
        print(pd.DataFrame(t).drop(columns=[c for c in t[0] if (c.startswith("tost") or c == "halfwidth_le_delta") == (y != "macro_f1") and c not in ("n_seeds", "runs", "se_total", "ci90_halfwidth")]).to_string(index=False))
        out[f"{y}_{dl}"] = dict(sd=sd, table=t)
    print(json.dumps({k: v["sd"] for k, v in out.items()}))
    print("NOTE: halfwidth_le_delta is necessary, not sufficient. Use tost_power_* (target >= 0.8).")

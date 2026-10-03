import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""Analyse pilot/sweep results: noise floor, layer x rank signal, reliability, power.

Input: results.csv from run_pilot.py (columns cell,seed,config,layer,rank,macro_f1,...).
Delta(m,r,s) = F1(layer m at rank r, seed s) - F1(baseline uniform r=8, same seed)  (paired).

Quantities (per rank r), with n_s seeds and L layers:
  layer means  d_m = mean_s Delta(m,r,s)
  Var_between  = Var_m(d_m)  = sigma2_LR + sigma2_eps / n_s
  sigma2_eps/n_s estimated by mean_m Var_s(Delta(m,r,.)) / n_s
  sigma2_LR    = max(0, Var_between - within)     (method of moments)
  rho          = sigma2_LR / Var_between          (reliability of the observed layer profile)
H1 test: permute layer labels within each seed (exchangeable under H0), statistic Var_between.
H2: split-half reliability (random halves of seeds), Spearman-Brown corrected.
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df.config == "baseline"].set_index("seed").macro_f1
    lay = df[df.config == "layer"].copy()
    lay["delta"] = lay.macro_f1 - lay.seed.map(base)
    if lay.delta.isna().any():
        raise ValueError("some seeds lack a baseline run")
    return lay


def cube(lay: pd.DataFrame, rank: int):
    """array [layers, seeds] of deltas at one rank"""
    sub = lay[lay["rank"] == rank]
    piv = sub.pivot(index="layer", columns="seed", values="delta").sort_index()
    return piv.to_numpy(), list(piv.index), list(piv.columns)


def variance_components(D: np.ndarray):
    L, S = D.shape
    d = D.mean(axis=1)
    var_between = d.var(ddof=1)
    within = D.var(axis=1, ddof=1).mean() / S
    s2_lr = max(0.0, var_between - within)
    rho = s2_lr / var_between if var_between > 0 else 0.0
    return {"var_between": var_between, "sigma2_eps_over_ns": within, "sigma2_LR": s2_lr, "rho": rho,
            "sigma2_eps": within * S}


def permutation_test(D: np.ndarray, B: int = 2000, rng=None):
    rng = rng or np.random.default_rng(0)
    L, S = D.shape
    obs = D.mean(axis=1).var(ddof=1)
    cnt = 0
    for _ in range(B):
        P = np.stack([rng.permutation(D[:, s]) for s in range(S)], axis=1)
        cnt += P.mean(axis=1).var(ddof=1) >= obs - 1e-15
    return (1 + cnt) / (1 + B)


def split_half(D: np.ndarray, B: int = 1000, rng=None):
    rng = rng or np.random.default_rng(1)
    L, S = D.shape
    if S < 4:
        return {"median": np.nan, "lo": np.nan, "hi": np.nan}
    vals = []
    for _ in range(B):
        idx = rng.permutation(S); a, b = idx[: S // 2], idx[S // 2 : 2 * (S // 2)]
        r = spearmanr(D[:, a].mean(1), D[:, b].mean(1)).correlation
        if not np.isnan(r):
            vals.append(2 * r / (1 + r) if r > -1 else -1)  # Spearman-Brown
    v = np.array(vals)
    return {"median": float(np.median(v)), "lo": float(np.quantile(v, .025)), "hi": float(np.quantile(v, .975))}


def simulate_power(sigma_lr: float, sigma_eps: float, n_s: int, L: int = 12, nsim: int = 500,
                   B: int = 199, alpha: float = 0.05, seed: int = 0):
    """Power of the H1 permutation test and mean reliability when the true layer effect has SD sigma_lr."""
    rng = np.random.default_rng(seed)
    hits, rhos = 0, []
    for _ in range(nsim):
        u = rng.normal(0, sigma_lr, L)
        D = u[:, None] + rng.normal(0, sigma_eps, (L, n_s))
        hits += permutation_test(D, B, rng) < alpha
        rhos.append(variance_components(D)["rho"])
    return {"power_H1": hits / nsim, "mean_rho": float(np.mean(rhos))}


def decide_G1(rho: float, p: float, hi=0.5, lo=0.2):
    """Provisional thresholds (to be confirmed by power simulation before the main grid)."""
    if rho >= hi and p < 0.05:
        return "GO: run main grid (C1-C4)"
    if rho >= lo:
        return "PARTIAL: run C1 and C3 only, more seeds, use AUC and coarser units"
    return "NO-GO at single-layer resolution: switch to coarse units (layer groups, module types) + equivalence/power"


def analyse(df: pd.DataFrame):
    lay = paired_deltas(df)
    out = {"n_seeds": int(lay.seed.nunique()), "by_rank": {}}
    for r in sorted(lay["rank"].unique()):
        D, layers, seeds = cube(lay, r)
        vc = variance_components(D)
        vc["perm_p"] = permutation_test(D)
        vc["split_half"] = split_half(D)
        vc["frac_layers_gt_2se"] = float(np.mean(np.abs(D.mean(1)) > 2 * D.std(1, ddof=1) / np.sqrt(D.shape[1])))
        vc["layer_means"] = {int(m): float(v) for m, v in zip(layers, D.mean(1))}
        out["by_rank"][int(r)] = vc
    if 1 in out["by_rank"]:
        vc = out["by_rank"][1]
        out["L_m_profile(rank=1)"] = {m: -v for m, v in vc["layer_means"].items()}
        out["G1_decision(rank=1)"] = decide_G1(vc["rho"], vc["perm_p"])
        out["power_if_truth_equals_estimate"] = simulate_power(
            np.sqrt(vc["sigma2_LR"]), np.sqrt(vc["sigma2_eps"]), out["n_seeds"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("results"); ap.add_argument("--cell", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.results)
    if a.cell:
        df = df[df.cell == a.cell]
    print(json.dumps(analyse(df), indent=2, default=float))

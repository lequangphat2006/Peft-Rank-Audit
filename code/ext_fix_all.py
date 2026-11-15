# code/ext_fix_all.py - Meta-analysis of the model x rank interaction (E1) across iterations.
# All numbers are read from the confirmatory*.json files (no hand-typed values).
# Reports (i) DerSimonian-Laird random effects with t(k-1) quantiles and
#         (ii) Hartung-Knapp-Sidik-Jonkman (HKSJ) interval, plus tau, Q and I^2.
# Also prints the per-iteration spec curve (total rank effect per model, 1 -> 32).
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

ITERATIONS = [
    ("B1X (main)", "results/ext/phobert/confirmatory_final_v4.json"),
    ("T2 (post-hoc, session-blocked)", "results/ext/t2/confirmatory_official.json"),
    ("T3-F2 (post-hoc, 4 seeds)", "results/ext/t3/confirmatory_T3_F2.json"),
    ("T3-F3 (post-hoc, 4 seeds)", "results/ext/t3/confirmatory_T3_F3.json"),
]
OUT_JSON = "results/ext/meta_analysis.json"


def load_e1(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        e1 = json.load(f)["E1"]
    return float(e1["diff"]), float(e1["se"]), float(e1["p_holm"])


def random_effects(deltas, ses):
    k = len(deltas)
    w = 1.0 / ses ** 2
    mu_fixed = np.sum(w * deltas) / np.sum(w)
    Q = float(np.sum(w * (deltas - mu_fixed) ** 2))
    C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (k - 1)) / C)
    w_re = 1.0 / (ses ** 2 + tau2)
    mu = float(np.sum(w_re * deltas) / np.sum(w_re))
    se_dl = float(np.sqrt(1.0 / np.sum(w_re)))
    t_crit = float(stats.t.ppf(0.975, df=k - 1))
    # Hartung-Knapp-Sidik-Jonkman: variance rescaled by the weighted residual mean square
    q_hk = float(np.sum(w_re * (deltas - mu) ** 2) / (k - 1))
    se_hk = float(np.sqrt(q_hk / np.sum(w_re)))
    se_hk_floor = float(np.sqrt(max(1.0, q_hk) / np.sum(w_re)))  # conservative variant (never narrower than DL)
    return {
        "k": k,
        "pooled": mu,
        "tau": float(np.sqrt(tau2)),
        "Q": Q,
        "Q_p": float(1 - stats.chi2.cdf(Q, k - 1)),
        "I2": float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0,
        "DL_z_ci95": [mu - 1.96 * se_dl, mu + 1.96 * se_dl],
        "DL_t_ci95": [mu - t_crit * se_dl, mu + t_crit * se_dl],
        "HKSJ_ci95": [mu - t_crit * se_hk, mu + t_crit * se_hk],
        "HKSJ_floor_ci95": [mu - t_crit * se_hk_floor, mu + t_crit * se_hk_floor],
    }


def main():
    rows = [(name,) + load_e1(p) for name, p in ITERATIONS]
    deltas = np.array([r[1] for r in rows])
    ses = np.array([r[2] for r in rows])
    print("=== E1 (PhoBERT - XLM-R rank slope) per iteration, read from JSON ===")
    for name, d, s, ph in rows:
        print(f"{name:34s} delta={d:+.4f}  se={s:.4f}  p_holm={ph:.3f}")
    res = random_effects(deltas, ses)
    print("\n=== META-ANALYSIS (4 non-independent iterations) ===")
    print(f"Pooled (random effects): {res['pooled']:+.4f}   tau={res['tau']:.4f}   "
          f"Q={res['Q']:.2f} (p={res['Q_p']:.3f})   I2={res['I2']:.0%}")
    print("DL + t(k-1) 95% CI : [{:+.4f}, {:+.4f}]".format(*res["DL_t_ci95"]))
    print("HKSJ 95% CI        : [{:+.4f}, {:+.4f}]".format(*res["HKSJ_ci95"]))
    print("HKSJ (floored) CI  : [{:+.4f}, {:+.4f}]".format(*res["HKSJ_floor_ci95"]))
    print("DL + z CI (not used in the paper; anti-conservative with k=4): "
          "[{:+.4f}, {:+.4f}]".format(*res["DL_z_ci95"]))
    print("Note: XLM-R data in B1X and T2 are identical (T2_XLM == R1), so the iterations are not independent.")
    out = {"iterations": [{"name": n, "delta": d, "se": s, "p_holm": p} for n, d, s, p in rows], **res}
    os.makedirs(os.path.dirname(os.path.join(ROOT, OUT_JSON)), exist_ok=True)
    with open(os.path.join(ROOT, OUT_JSON), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten: {OUT_JSON}")
    spec_curve()


def spec_curve():
    """Total rank effect (rank 1 -> 32, per-seed mean) for each model in B1X and T2."""
    try:
        from ext_common import read_csv, load_lr_map, select_branch, per_seed_effects
    except Exception as e:  # pragma: no cover
        print(f"\n[spec curve skipped: {e}]")
        return
    sets = {
        "B1X": ("results/ext/phobert/B1X_total.csv", "results/ext/xlmr/XLM_total_v2.csv"),
        "T2": ("results/ext/t2/T2_PHO.csv", "results/ext/t2/T2_XLM.csv"),
    }
    print("\n=== SPEC CURVE: total effect rank 1 -> 32 (macro_f1, LR map branch) ===")
    for name, (pf, xf) in sets.items():
        vals = []
        for f, m in ((pf, "results/ext/phobert/lr_map_best.json"), (xf, "results/ext/xlmr/lr_map_best.json")):
            df = read_csv(os.path.join(ROOT, f))
            df = df[df["epochs"] == 6]
            sp, _ = per_seed_effects(select_branch(df, load_lr_map(os.path.join(ROOT, m))), "macro_f1",
                                     ranks=[1, 4, 8, 16, 32])
            vals.append((sp["total"].mean(), sp["total"].std(ddof=1), len(sp)))
        (mp, sp_, np_), (mx, sx, nx) = vals
        print(f"{name:4s} PhoBERT {mp:+.4f} (SD {sp_:.4f}, n={np_}) | XLM-R {mx:+.4f} (SD {sx:.4f}, n={nx}) | "
              f"P - X = {mp - mx:+.4f}")


if __name__ == "__main__":
    main()

"""Stage 1 analysis (pre-registered).

Tests:
- H1a: attention.self.key, seeds 105-109, paired within-seed delta
- H1b: intermediate.dense, seeds 105-109, paired within-seed delta
- Holm correction m=2

Seeds 100-104 are for hypothesis generation only, not used for tests.

Estimates:
- tau^2 = variance of true module effects (bootstrap CI)
- per-module paired delta mean, std, t, p
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse, json
import numpy as np
import pandas as pd
from scipy import stats


H1_MODULES = ['attention.self.key', 'intermediate.dense']
TEST_SEEDS = [105, 106, 107, 108, 109]
ALL_MODULES = [
    'attention.self.query',
    'attention.self.key',
    'attention.self.value',
    'attention.output.dense',
    'intermediate.dense',
    'output.dense',
]


def holm(pvals, m=None):
    """Holm-Bonferroni correction. Return adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    m = m if m is not None else len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    for i, idx in enumerate(order):
        adj[idx] = min(1.0, pvals[idx] * (m - i))
    # Enforce monotonicity
    for i in range(1, m):
        if adj[order[i]] < adj[order[i-1]]:
            adj[order[i]] = adj[order[i-1]]
    return adj


def paired_delta(df, module, seeds):
    """Return array of paired within-seed delta (rank32 - rank1)."""
    sub = df[(df.module == module) & (df.seed.isin(seeds))]
    piv = sub.pivot_table(index='seed', columns='rank',
                          values='macro_f1')
    piv = piv.dropna()
    if len(piv) == 0 or 1 not in piv.columns or 32 not in piv.columns:
        return np.array([])
    return (piv[32] - piv[1]).values


def bootstrap_tau2(df, seeds, n_boot=2000, rng_seed=0):
    """Bootstrap tau^2 = var of true module means.
    
    tau^2 = Var(module_means) - mean(Var(paired_delta)/n_seeds)
    """
    rng = np.random.default_rng(rng_seed)
    seeds_arr = np.array(seeds)
    n = len(seeds_arr)
    tau2_samples = []

    for _ in range(n_boot):
        resampled = rng.choice(seeds_arr, size=n, replace=True)
        module_means = []
        module_vars = []
        for mod in ALL_MODULES:
            d = paired_delta(df, mod, resampled)
            if len(d) < 2:
                continue
            module_means.append(d.mean())
            module_vars.append(d.var(ddof=1))
        if len(module_means) < 3:
            continue
        var_between = np.var(module_means, ddof=1)
        noise = np.mean(module_vars) / n
        tau2 = max(0.0, var_between - noise)
        tau2_samples.append(tau2)
    tau2_samples = np.array(tau2_samples)
    return {
        'mean': float(tau2_samples.mean()),
        'lo': float(np.percentile(tau2_samples, 2.5)),
        'hi': float(np.percentile(tau2_samples, 97.5)),
        'n_valid': int(len(tau2_samples)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--out', default=None)
    ap.add_argument('--use_seeds', type=int, nargs='*', default=TEST_SEEDS)
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    df = df[df.config.str.startswith('type:')].copy()
    df['module'] = df['config'].apply(lambda x: x.split(':', 1)[1])

    out = {'test_seeds': args.use_seeds}

    # === Per-module paired delta ===
    print('=' * 70)
    print(f'Per-module paired delta (rank32 - rank1), seeds {args.use_seeds}')
    print('=' * 70)
    print(f'{"module":30s}  {"mean":>10s}  {"std":>10s}  {"t":>8s}  {"p":>8s}  {"n":>3s}')
    print('-' * 76)
    per_module = {}
    for mod in ALL_MODULES:
        d = paired_delta(df, mod, args.use_seeds)
        if len(d) < 2:
            print(f'{mod:30s}  (insufficient data)')
            continue
        m, s = d.mean(), d.std(ddof=1)
        t, p = stats.ttest_1samp(d, 0)
        per_module[mod] = {'mean': float(m), 'std': float(s),
                           't': float(t), 'p': float(p), 'n': len(d)}
        print(f'{mod:30s}  {m:>+10.5f}  {s:>10.5f}  {t:>+8.2f}  {p:>8.4f}  {len(d):>3d}')
    out['per_module'] = per_module

    # === H1 tests + Holm ===
    print()
    print('=' * 70)
    print('Confirmatory H1 tests (Holm m=2)')
    print('=' * 70)
    pvals = [per_module[m]['p'] for m in H1_MODULES if m in per_module]
    adj = holm(pvals, m=2)
    h1_out = {}
    for i, mod in enumerate(H1_MODULES):
        if mod not in per_module:
            continue
        p_adj = adj[i]
        sig = p_adj < 0.05
        h1_out[mod] = {'p_raw': per_module[mod]['p'],
                       'p_holm': float(p_adj),
                       'significant': bool(sig)}
        print(f'{mod:30s}  p_raw={per_module[mod]["p"]:.4f}  '
              f'p_holm={p_adj:.4f}  {"SIG" if sig else "n.s."}')
    out['H1'] = h1_out

    # === tau^2 via bootstrap ===
    print()
    print('=' * 70)
    print(f'Bootstrap tau^2 (true variance of module effects)')
    print('=' * 70)
    tau2 = bootstrap_tau2(df, args.use_seeds, n_boot=2000, rng_seed=0)
    print(f'tau^2 mean = {tau2["mean"]:.6f}  '
          f'CI95 = [{tau2["lo"]:.6f}, {tau2["hi"]:.6f}]  '
          f'(n_valid={tau2["n_valid"]})')
    tau = np.sqrt(max(0, tau2['mean']))
    print(f'tau (std) = {tau:.5f}')
    out['tau2'] = tau2
    out['tau'] = float(tau)

    # === Decision rule (pre-registered) ===
    print()
    print('=' * 70)
    print('Decision rule (pre-registered)')
    print('=' * 70)
    lower = tau2['lo']
    any_h1_sig = any(v['significant'] for v in h1_out.values())
    if any_h1_sig and lower >= 0.002:
        decision = 'STAGE_2'
        msg = 'H1 significant + tau2 lower CI >= 0.002 -> run Stage 2 (10 more seeds)'
    elif lower < 0.001:
        decision = 'NEGATIVE'
        msg = 'tau2 lower CI < 0.001 -> negative result, equivalence test'
    else:
        decision = 'INCONCLUSIVE'
        msg = 'tau2 CI too wide -> collect more seeds or refine design'
    print(f'Decision: {decision}')
    print(f'Reason: {msg}')
    out['decision'] = decision
    out['decision_reason'] = msg

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
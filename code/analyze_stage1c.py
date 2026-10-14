"""Analyze Stage 1c — full dose-response 6 modules × 4 ranks × 5 seeds."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse, json
import numpy as np
import pandas as pd
from scipy import stats


MODULES = [
    'attention.self.query',
    'attention.self.key',
    'attention.self.value',
    'attention.output.dense',
    'intermediate.dense',
    'output.dense',
]
RANKS = [1, 4, 16, 32]


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    for i, idx in enumerate(order):
        adj[idx] = min(1.0, p[idx] * (m - i))
    for i in range(1, m):
        if adj[order[i]] < adj[order[i-1]]:
            adj[order[i]] = adj[order[i-1]]
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    df['module'] = df['config'].apply(lambda x: x.split(':', 1)[1])

    out = {'modules': MODULES, 'ranks': RANKS}

    # === 1. Mean F1 ===
    print('=' * 80)
    print('Mean Macro-F1 by (module, rank), 5 seeds')
    print('=' * 80)
    piv_f1 = df.pivot_table(index='module', columns='rank',
                            values='macro_f1', aggfunc='mean')
    print(piv_f1.round(4).to_string())

    # === 2. Std F1 ===
    print('\n' + '=' * 80)
    print('Std Macro-F1 by (module, rank)')
    print('=' * 80)
    piv_std = df.pivot_table(index='module', columns='rank',
                             values='macro_f1', aggfunc='std')
    print(piv_std.round(4).to_string())

    # === 3. Paired delta vs rank 1 ===
    print('\n' + '=' * 80)
    print('Paired delta vs rank 1 (mean +/- std across 5 seeds)')
    print('=' * 80)
    header = f'{"module":30s} | ' + ' | '.join(
        f'{"rank " + str(r):>16s}' for r in RANKS[1:])
    print(header)
    print('-' * 90)

    delta_summary = {}
    for mod in MODULES:
        sub = df[df.module == mod]
        piv = sub.pivot_table(index='seed', columns='rank',
                              values='macro_f1')
        row = [f'{mod:30s}']
        mod_deltas = {}
        for r in RANKS[1:]:
            d = (piv[r] - piv[1]).values
            m, s = d.mean(), d.std(ddof=1)
            row.append(f'{m:+.4f}+/-{s:.4f}')
            mod_deltas[str(r)] = {'mean': float(m), 'std': float(s),
                                  'per_seed': d.tolist()}
        print(' | '.join(row))
        delta_summary[mod] = mod_deltas
    out['delta'] = delta_summary

    # === 4. Spearman rho (dose-response) ===
    print('\n' + '=' * 80)
    print('Spearman rho (rank vs F1), 20 points (5 seeds x 4 ranks)')
    print('=' * 80)
    print(f'{"module":30s} | {"rho":>8s} | {"p_raw":>8s} | {"direction":>10s}')
    print('-' * 70)
    rho_summary = {}
    pvals = []
    for mod in MODULES:
        sub = df[df.module == mod]
        rho, p = stats.spearmanr(sub['rank'], sub['macro_f1'])
        direction = 'increasing' if rho > 0 else 'decreasing'
        print(f'{mod:30s} | {rho:>+8.3f} | {p:>8.4f} | {direction:>10s}')
        rho_summary[mod] = {'rho': float(rho), 'p': float(p)}
        pvals.append(p)
    out['rho'] = rho_summary

    adj = holm(pvals)
    print('\nHolm corrected (m=6):')
    for i, mod in enumerate(MODULES):
        sig = 'SIG' if adj[i] < 0.05 else 'n.s.'
        print(f'  {mod:30s} p_raw={pvals[i]:.4f} p_holm={adj[i]:.4f} {sig}')

    # === 5. Paired t-test rank 32 vs rank 1 ===
    print('\n' + '=' * 80)
    print('Paired t-test: delta(rank 32 - rank 1) vs 0')
    print('=' * 80)
    print(f'{"module":30s} | {"mean":>9s} | {"t":>8s} | {"p_raw":>8s}')
    print('-' * 70)
    ttest_summary = {}
    pvals_t = []
    for mod in MODULES:
        sub = df[df.module == mod]
        piv = sub.pivot_table(index='seed', columns='rank',
                              values='macro_f1')
        d = (piv[32] - piv[1]).values
        t, p = stats.ttest_1samp(d, 0)
        print(f'{mod:30s} | {d.mean():>+9.4f} | {t:>+8.2f} | {p:>8.4f}')
        ttest_summary[mod] = {'mean': float(d.mean()),
                              't': float(t), 'p': float(p)}
        pvals_t.append(p)
    out['ttest_rank32'] = ttest_summary

    adj_t = holm(pvals_t)
    print('\nHolm corrected (m=6):')
    for i, mod in enumerate(MODULES):
        sig = 'SIG' if adj_t[i] < 0.05 else 'n.s.'
        print(f'  {mod:30s} p_raw={pvals_t[i]:.4f} p_holm={adj_t[i]:.4f} {sig}')

    # === 6. Attention vs FFN ===
    print('\n' + '=' * 80)
    print('Attention vs FFN (delta rank 32 - rank 1)')
    print('=' * 80)
    attn = ['attention.self.query', 'attention.self.key',
            'attention.self.value', 'attention.output.dense']
    ffn = ['intermediate.dense', 'output.dense']
    df['family'] = df.module.apply(
        lambda m: 'attention' if m in attn else 'ffn')
    for fam, mods in [('attention', attn), ('ffn', ffn)]:
        deltas = []
        for mod in mods:
            sub = df[df.module == mod]
            piv = sub.pivot_table(index='seed', columns='rank',
                                  values='macro_f1')
            deltas.extend((piv[32] - piv[1]).values)
        deltas = np.array(deltas)
        m, s = deltas.mean(), deltas.std(ddof=1)
        print(f'{fam:12s}: mean={m:+.5f}, std={s:.5f}, n={len(deltas)}')

    # === 7. Best rank per module ===
    print('\n' + '=' * 80)
    print('Best rank per module (max mean F1)')
    print('=' * 80)
    best = {}
    for mod in MODULES:
        sub = df[df.module == mod]
        mean_by_rank = sub.groupby('rank')['macro_f1'].mean()
        best_r = int(mean_by_rng.idxmax()) if False else int(mean_by_rank.idxmax())
        best_v = float(mean_by_rank.max())
        baseline_v = float(mean_by_rank[1])
        gap = best_v - baseline_v
        print(f'{mod:30s} best_rank={best_r:>2d}  F1={best_v:.4f}  '
              f'baseline(rank1)={baseline_v:.4f}  gap={gap:+.4f}')
        best[mod] = {'best_rank': best_r, 'best_f1': best_v,
                     'baseline_f1': baseline_v, 'gap': gap}
    out['best_rank'] = best

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
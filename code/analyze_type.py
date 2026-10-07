"""Analyze module-type pilot — paired within-seed delta (rank32 - rank1)."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse, json, os
import numpy as np
import pandas as pd
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    df['module'] = df['config'].apply(
        lambda x: x.split(':', 1)[1] if x.startswith('type:') else None)

    # Pivot: rows=(module, seed), cols=rank
    typ = df[df.config.str.startswith('type:')]
    piv = typ.pivot_table(
        index=['module', 'seed'], columns='rank', values='macro_f1'
    ).reset_index()
    piv['pair_delta'] = piv[32] - piv[1]   # rank32 - rank1, cùng seed

    print('=' * 70)
    print('Paired within-seed delta = F1(rank32) - F1(rank1)')
    print('=' * 70)
    summary = piv.groupby('module')['pair_delta'].agg(
        ['mean', 'std', 'count']).round(5)
    print(summary)

    print('\n' + '=' * 70)
    print('Per-module ICC (rank effect, paired)')
    print('=' * 70)
    icc_out = {}
    for mod in piv.module.unique():
        sub = piv[piv.module == mod]['pair_delta'].values
        # Single mean vs 0 — không có "between" vì 1 module 1 delta
        # Nhưng có thể test: is mean(pair_delta) != 0?
        t, p = stats.ttest_1samp(sub, 0) if len(sub) > 1 else (np.nan, np.nan)
        icc_out[mod] = {
            'mean': float(np.mean(sub)),
            'std': float(np.std(sub, ddof=1)),
            't': float(t) if not np.isnan(t) else None,
            'p': float(p) if not np.isnan(p) else None,
        }
        print(f'{mod:35s}  mean={np.mean(sub):+.5f}  std={np.std(sub, ddof=1):.5f}  '
              f't={t:+.2f}  p={p:.4f}')

    # ICC across modules: var_between_modules / (var_between + var_within)
    print('\n' + '=' * 70)
    print('ICC across modules (does module change ΔF1?)')
    print('=' * 70)
    module_means = piv.groupby('module')['pair_delta'].mean().values
    module_vars = piv.groupby('module')['pair_delta'].var(ddof=1).values
    var_between = float(np.var(module_means, ddof=1))
    var_within = float(np.mean(module_vars))
    icc = var_between / (var_between + var_within) if (var_between + var_within) > 0 else 0.0
    print(f'var_between={var_between:.6f}  var_within={var_within:.6f}  ICC={icc:.3f}')

    # Attention vs FFN
    print('\n' + '=' * 70)
    print('Attention vs FFN (paired delta)')
    print('=' * 70)
    attn = ['attention.self.query', 'attention.self.key',
            'attention.self.value', 'attention.output.dense']
    ffn = ['intermediate.dense', 'output.dense']
    piv['family'] = piv.module.apply(
        lambda m: 'attention' if m in attn else ('ffn' if m in ffn else 'other'))
    fam = piv.groupby('family')['pair_delta'].agg(['mean', 'std', 'count']).round(5)
    print(fam)

    # Save
    if args.out:
        out = {
            'paired_summary': summary.reset_index().to_dict(orient='records'),
            'icc_per_module': icc_out,
            'icc_across_modules': icc,
            'var_between': var_between,
            'var_within': var_within,
            'family_summary': fam.reset_index().to_dict(orient='records'),
        }
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
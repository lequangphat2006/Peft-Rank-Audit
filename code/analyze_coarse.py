"""Analyze pilot results at coarse-unit resolution (layer groups).

Aggregates single-layer sweep into 3 layer groups:
- bottom: layers 0-3
- middle: layers 4-7
- top:    layers 8-11

Computes per-group ΔF1 (vs baseline) and tests if group explains variance.
"""
import argparse
import json
import numpy as np
import pandas as pd
from scipy import stats


def group_of(layer):
    if layer < 4: return 'bottom'
    if layer < 8: return 'middle'
    return 'top'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--cell', default='C1')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    df = df[df.cell == args.cell]

    # Baseline per seed
    base = df[df.config == 'baseline'][['seed', 'macro_f1']].rename(
        columns={'macro_f1': 'f1_base'})

    # Layer runs
    lay = df[df.config == 'layer'].merge(base, on='seed')
    lay['delta'] = lay['macro_f1'] - lay['f1_base']
    lay['group'] = lay['layer'].apply(group_of)

    out = {'cell': args.cell, 'n_seeds': int(lay.seed.nunique())}

    # === Per-group per-rank summary ===
    print('=' * 60)
    print('Group-level ΔF1 (vs baseline), mean ± std across seeds')
    print('=' * 60)
    summary = lay.groupby(['group', 'rank'])['delta'].agg(
        ['mean', 'std', 'count']).round(5)
    print(summary)

    out['group_summary'] = summary.reset_index().to_dict(orient='records')

    # === ANOVA: does group explain ΔF1? ===
    print('\n' + '=' * 60)
    print('ANOVA: group effect on ΔF1 (per rank)')
    print('=' * 60)
    anova = {}
    for rank in [1, 32]:
        sub = lay[lay['rank'] == rank]
        groups = [sub[sub.group == g]['delta'].values
                  for g in ['bottom', 'middle', 'top']]
        f, p = stats.f_oneway(*groups)
        print(f'Rank {rank}: F = {f:.3f}, p = {p:.4f}')
        anova[str(rank)] = {'F': float(f), 'p': float(p)}
    out['anova_group_effect'] = anova

    # === Per-seed group mean (để xem stability) ===
    print('\n' + '=' * 60)
    print('Per-seed group mean ΔF1')
    print('=' * 60)
    per_seed = lay.groupby(['group', 'rank', 'seed'])['delta'].mean().reset_index()
    pivot = per_seed.pivot_table(index='seed', columns=['group', 'rank'],
                                 values='delta').round(5)
    print(pivot)

    # === Group-level variance decomposition (rough ICC) ===
    print('\n' + '=' * 60)
    print('Group-level reliability (rough)')
    print('=' * 60)
    icc_out = {}
    for rank in [1, 32]:
        sub = lay[lay['rank'] == rank]
        # 3 groups × n_seeds values per group
        group_means = sub.groupby(['group', 'seed'])['delta'].mean().unstack('seed')
        # group_means shape: (3 groups, n_seeds)
        grand_mean = group_means.values.mean()
        # Between-group variance
        gm = group_means.mean(axis=1).values  # per group
        var_between = float(np.var(gm, ddof=1))
        # Within-group (seed) variance
        var_within = float(group_means.values.var(axis=1, ddof=1).mean())
        icc = var_between / (var_between + var_within) if (var_between + var_within) > 0 else 0.0
        print(f'Rank {rank}: var_between={var_between:.6f}, '
              f'var_within={var_within:.6f}, ICC={icc:.3f}')
        icc_out[str(rank)] = {'var_between': var_between,
                              'var_within': var_within,
                              'icc': icc}
    out['icc'] = icc_out

    # Save JSON
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
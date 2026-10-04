"""Analyze module-type pilot results.

Reads results_type.csv with config = 'type:MODULE_NAME'.
Parses module name, computes ΔF1 vs baseline, ICC across modules.
"""
import argparse
import json
import numpy as np
import pandas as pd
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--cell', default='C1')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    df = df[df.cell == args.cell]

    # Parse module type từ config 'type:MODULE_NAME'
    df['module'] = df['config'].apply(
        lambda x: x.split(':', 1)[1] if x.startswith('type:') else None
    )

    # Baseline — lấy từ results_C1.csv nếu results_type.csv không có baseline
    if (df.config == 'baseline').any():
        base = df[df.config == 'baseline'][['seed', 'macro_f1']].rename(
            columns={'macro_f1': 'f1_base'})
    else:
        import os
        # Tìm results_C1.csv cùng thư mục
        base_path = os.path.join(os.path.dirname(args.results), 'results_C1.csv')
        if not os.path.exists(base_path):
            raise FileNotFoundError(
                f'Không tìm thấy baseline. Cần results_C1.csv ở {base_path}')
        base_df = pd.read_csv(base_path)
        base = base_df[base_df.config == 'baseline'][['seed', 'macro_f1']].rename(
            columns={'macro_f1': 'f1_base'})
        print(f'Baseline loaded from: {base_path}')
        print(base)

    # Type runs
    typ = df[df.config.str.startswith('type:')].merge(base, on='seed')
    typ['delta'] = typ['macro_f1'] - typ['f1_base']

    out = {'cell': args.cell, 'n_seeds': int(typ.seed.nunique())}

    # === Per-module summary ===
    print('=' * 70)
    print('Module-level ΔF1 (vs baseline), mean ± std across seeds')
    print('=' * 70)
    summary = typ.groupby(['module', 'rank'])['delta'].agg(
        ['mean', 'std', 'count']).round(5)
    print(summary)
    out['module_summary'] = summary.reset_index().to_dict(orient='records')

    # === ICC per module (rank 1 vs 32 as 'conditions') ===
    print('\n' + '=' * 70)
    print('Per-module ICC: does rank matter within module?')
    print('=' * 70)
    icc_out = {}
    for mod in typ.module.unique():
        sub = typ[typ.module == mod]
        # Pivot: rows=seed, cols=rank
        pivot = sub.pivot_table(index='seed', columns='rank',
                                values='delta', aggfunc='mean')
        if pivot.shape[1] < 2:
            continue
        # ICC approx: between-rank var vs within (seed) var
        # Use simple formula: var of rank means / (var of rank means + mean var across seeds)
        rank_means = pivot.mean(axis=0).values
        seed_vars = pivot.var(axis=0, ddof=1).values
        var_between = float(np.var(rank_means, ddof=1)) if len(rank_means) > 1 else 0.0
        var_within = float(np.mean(seed_vars))
        icc = var_between / (var_between + var_within) if (var_between + var_within) > 0 else 0.0
        icc_out[mod] = {'var_between': var_between, 'var_within': var_within,
                        'icc': icc,
                        'rank1_mean': float(rank_means[0]),
                        'rank32_mean': float(rank_means[1]) if len(rank_means) > 1 else None}
        print(f'{mod:35s}  ICC={icc:.3f}  rank1={rank_means[0]:+.4f}  rank32={rank_means[1]:+.4f}')
    out['icc_per_module'] = icc_out

    # === Attention vs FFN aggregation ===
    print('\n' + '=' * 70)
    print('Attention vs FFN')
    print('=' * 70)
    attn = ['attention.self.query', 'attention.self.key',
            'attention.self.value', 'attention.output.dense']
    ffn = ['intermediate.dense', 'output.dense']
    typ['family'] = typ.module.apply(
        lambda m: 'attention' if m in attn else ('ffn' if m in ffn else 'other'))
    fam = typ.groupby(['family', 'rank'])['delta'].agg(['mean', 'std']).round(5)
    print(fam)
    out['family_summary'] = fam.reset_index().to_dict(orient='records')

    # === ANOVA: family effect ===
    print('\n' + '=' * 70)
    print('ANOVA: family × ΔF1 (per rank)')
    print('=' * 70)
    anova_out = {}
    for rank in [1, 32]:
        sub = typ[typ['rank'] == rank]
        groups = [sub[sub.family == f]['delta'].values
                  for f in ['attention', 'ffn'] if len(sub[sub.family == f]) > 0]
        if len(groups) == 2:
            f, p = stats.f_oneway(*groups)
            print(f'Rank {rank}: F={f:.3f}, p={p:.4f}')
            anova_out[str(rank)] = {'F': float(f), 'p': float(p)}
    out['anova_family'] = anova_out

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
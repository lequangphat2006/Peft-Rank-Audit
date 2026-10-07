"""Stage 1a analysis — dose-response test.

2 modules × 4 ranks × 2 seeds.
Purpose: check if F1 vs rank is monotone (signal) or chaotic (noise).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse, json
import numpy as np
import pandas as pd
from scipy import stats

MODULES = ['attention.self.key', 'intermediate.dense']
RANKS = [1, 4, 16, 32]
SEEDS = [100, 105]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    df = df[df.config.str.startswith('type:')].copy()
    df['module'] = df['config'].apply(lambda x: x.split(':', 1)[1])

    out = {'modules': MODULES, 'ranks': RANKS, 'seeds': SEEDS}

    # === Raw table: F1 and NLL ===
    print('=' * 80)
    print('Raw: Macro-F1 by (module, rank, seed)')
    print('=' * 80)
    for mod in MODULES:
        sub = df[df.module == mod].sort_values(['rank', 'seed'])
        print(f'\n>>> {mod}')
        print(f'{"rank":>4} | ' + ' | '.join(f'seed {s}' for s in SEEDS))
        for r in RANKS:
            row = sub[sub['rank'] == r]
            vals = [row[row.seed == s]['macro_f1'].values for s in SEEDS]
            line = f'{r:>4} | ' + ' | '.join(
                f'{v[0]:.4f}' if len(v) > 0 else '   --'
                for v in vals
            )
            print(line)

    print('\n' + '=' * 80)
    print('Raw: NLL by (module, rank, seed)')
    print('=' * 80)
    for mod in MODULES:
        sub = df[df.module == mod].sort_values(['rank', 'seed'])
        print(f'\n>>> {mod}')
        print(f'{"rank":>4} | ' + ' | '.join(f'seed {s}' for s in SEEDS))
        for r in RANKS:
            row = sub[sub['rank'] == r]
            vals = [row[row.seed == s]['nll'].values for s in SEEDS]
            line = f'{r:>4} | ' + ' | '.join(
                f'{v[0]:.4f}' if len(v) > 0 else '   --'
                for v in vals
            )
            print(line)

    # === Paired delta vs rank 1 (within seed) ===
    print('\n' + '=' * 80)
    print('Paired delta vs rank 1 (per seed, mean ± std)')
    print('=' * 80)
    trend_out = {}
    for mod in MODULES:
        sub = df[df.module == mod]
        piv = sub.pivot_table(index='seed', columns='rank',
                              values='macro_f1')
        if 1 not in piv.columns:
            continue
        print(f'\n>>> {mod}')
        print(f'{"rank":>4} | {"delta mean":>12} | {"delta std":>12} | ' +
              ' | '.join(f'seed {s}' for s in SEEDS))
        mod_trend = {'ranks': RANKS, 'delta_mean': [], 'delta_std': []}
        for r in RANKS:
            if r not in piv.columns:
                continue
            delta = piv[r] - piv[1]
            m, s = delta.mean(), delta.std(ddof=1) if len(delta) > 1 else 0.0
            mod_trend['delta_mean'].append(float(m))
            mod_trend['delta_std'].append(float(s))
            seed_vals = ' | '.join(
                f'{delta[s]:+.4f}' if s in delta.index else '   --'
                for s in SEEDS
            )
            print(f'{r:>4} | {m:>+12.5f} | {s:>12.5f} | {seed_vals}')
        trend_out[mod] = mod_trend

    # === Monotonicity check ===
    print('\n' + '=' * 80)
    print('Monotonicity check (Spearman rho between rank and F1)')
    print('=' * 80)
    mono_out = {}
    for mod in MODULES:
        sub = df[df.module == mod]
        # Combine across seeds (2 seeds × 4 ranks = 8 points)
        rho, p = stats.spearmanr(sub['rank'], sub['macro_f1'])
        # Direction per seed
        rho_per_seed = {}
        for s in SEEDS:
            ss = sub[sub.seed == s]
            if len(ss) >= 3:
                rho_s, _ = stats.spearmanr(ss['rank'], ss['macro_f1'])
                rho_per_seed[s] = float(rho_s)
        mono_out[mod] = {
            'rho_all': float(rho),
            'p_all': float(p),
            'rho_per_seed': rho_per_seed,
        }
        print(f'{mod:35s} rho={rho:+.3f}  p={p:.4f}  '
              f'per-seed={ {k: round(v,3) for k,v in rho_per_seed.items()} }')

    # === Summary judgement ===
    print('\n' + '=' * 80)
    print('Judgement (rule: monotone if |rho| > 0.7 and same sign across seeds)')
    print('=' * 80)
    judgement = {}
    for mod in MODULES:
        info = mono_out[mod]
        same_sign = (len(info['rho_per_seed']) >= 2 and
                     all(np.sign(v) == np.sign(list(info['rho_per_seed'].values())[0])
                         for v in info['rho_per_seed'].values()))
        monotone = abs(info['rho_all']) > 0.7 and same_sign
        judgement[mod] = {
            'monotone': monotone,
            'rho_all': info['rho_all'],
            'same_sign': same_sign,
        }
        verdict = 'MONOTONE (signal)' if monotone else 'CHAOTIC (noise)'
        print(f'{mod:35s} -> {verdict}')

    out['trend'] = trend_out
    out['mono'] = mono_out
    out['judgement'] = judgement

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
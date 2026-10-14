"""Merge Stage 1c S1 + S2 + Stage 1b.

Filter to seeds 105-109 to keep balanced design (5 seeds per cell).

Sources:
- stage1c_s1_results.csv: 6 modules × 2 ranks {1, 16} × 5 seeds = 60 jobs
- stage1c_s2_results.csv: 6 modules × 2 ranks {4, 32} × 5 seeds = 60 jobs
- stage1b_results.csv: intermediate.dense × 2 ranks {1, 32} × seeds {100, 105}

After filter: 6 modules × 4 ranks {1, 4, 16, 32} × 5 seeds {105-109} = 120 rows.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

BASE = 'results/pilot'
SEEDS_KEEP = [105, 106, 107, 108, 109]
EXPECTED_MODULES = [
    'attention.self.query',
    'attention.self.key',
    'attention.self.value',
    'attention.output.dense',
    'intermediate.dense',
    'output.dense',
]
EXPECTED_RANKS = [1, 4, 16, 32]


def main():
    # === Load 3 files ===
    df_s1 = pd.read_csv(f'{BASE}/stage1c_s1_results.csv')
    df_s2 = pd.read_csv(f'{BASE}/stage1c_s2_results.csv')
    df_1b = pd.read_csv(f'{BASE}/stage1b_results.csv')

    print(f'Loaded S1: {len(df_s1)} rows')
    print(f'Loaded S2: {len(df_s2)} rows')
    print(f'Loaded 1b: {len(df_1b)} rows')

    # === Merge ===
    df = pd.concat([df_s1, df_s2, df_1b], ignore_index=True)

    # Drop duplicates (same seed, config, layer, rank)
    df = df.drop_duplicates(
        subset=['seed', 'config', 'layer', 'rank'], keep='first')

    # === Filter to seeds 105-109 for balanced design ===
    df = df[df.seed.isin(SEEDS_KEEP)].copy()
    df = df.sort_values(['seed', 'config', 'rank']).reset_index(drop=True)

    # === Parse module name ===
    df['module'] = df['config'].apply(
        lambda x: x.split(':', 1)[1] if x.startswith('type:') else None)

    # === Verify design ===
    counts = df.groupby(['module', 'rank']).size()

    print()
    print('=' * 70)
    print('=== Count by (module, rank) ===')
    print('=' * 70)
    print(counts.to_string())

    # Expected: 6 modules × 4 ranks = 24 cells, mỗi cell 5 seeds
    expected_cells = len(EXPECTED_MODULES) * len(EXPECTED_RANKS)
    actual_cells = len(counts)

    print()
    print(f'Expected cells: {expected_cells}')
    print(f'Actual cells:   {actual_cells}')
    print(f'Expected rows:  {expected_cells * 5}')
    print(f'Actual rows:    {len(df)}')

    # Assert balance
    unbalanced = counts[counts != 5]
    if len(unbalanced) > 0:
        print()
        print('WARNING: Unbalanced cells:')
        print(unbalanced.to_string())
    else:
        print()
        print('Balanced design: OK (mỗi cell = 5 seeds)')

    # === Check all modules/ranks present ===
    modules_in = sorted(df.module.unique())
    ranks_in = sorted(df['rank'].unique())

    print()
    print(f'Modules present: {modules_in}')
    print(f'Ranks present:   {ranks_in}')

    missing_mod = set(EXPECTED_MODULES) - set(modules_in)
    missing_rank = set(EXPECTED_RANKS) - set(ranks_in)
    if missing_mod:
        print(f'WARNING: Missing modules: {missing_mod}')
    if missing_rank:
        print(f'WARNING: Missing ranks: {missing_rank}')

    # === Save ===
    out = f'{BASE}/stage1c_merged.csv'
    df.to_csv(out, index=False)
    print()
    print(f'Saved: {out} ({len(df)} rows)')

    # === Quick preview ===
    print()
    print('=' * 70)
    print('=== Preview: first 8 rows ===')
    print('=' * 70)
    print(df[['seed', 'module', 'rank', 'macro_f1', 'nll']].head(8).to_string())


if __name__ == '__main__':
    main()
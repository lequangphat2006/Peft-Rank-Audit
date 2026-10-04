"""Power analysis cho negative result pilot.

Tính power nếu effect size thực sự là 0.005, 0.010, 0.015 với n_seeds = 5, 10, 20, 30.
Dùng two-sample t-test giữa rank 1 vs rank 32 (paired by seed).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestPower


def main():
    df = pd.read_csv('results/pilot/results_type.csv')
    df['module'] = df['config'].apply(lambda x: x.split(':', 1)[1])

    # ΔF1 (rank 32 - rank 1) per seed per module
    piv = df.pivot_table(index=['module', 'seed'], columns='rank',
                         values='macro_f1').reset_index()
    piv['delta'] = piv[32] - piv[1]

    # Observed std of delta
    std_delta = piv.groupby('module')['delta'].std().mean()
    print(f'Observed std of (rank32 - rank1) delta: {std_delta:.5f}')
    print()

    # Power analysis
    alpha = 0.05
    powers = {}
    for effect in [0.005, 0.010, 0.015, 0.020]:
        # TTestPower.solve_power uses Cohen's d
        d = effect / std_delta
        for n in [5, 10, 20, 30, 50]:
            p = TTestPower().power(effect_size=d, nobs=n,
                                   alpha=alpha, alternative='two-sided')
            powers.setdefault(effect, {})[n] = round(p, 3)

    print('=== Power vs n_seeds ===')
    print(f'{"effect":>8} | ' + ' | '.join(f'n={n:>3}' for n in [5, 10, 20, 30, 50]))
    print('-' * 60)
    for eff, row in powers.items():
        line = f'{eff:>8.3f} | ' + ' | '.join(f'{row[n]:>5.3f}' for n in [5, 10, 20, 30, 50])
        print(line)

    print()
    print('=== Đọc ===')
    print('- Power >= 0.8 là ngưỡng chuẩn')
    print('- Nếu power < 0.8 ngay cả ở n=50 → effect cần lớn hơn để detect')
    print('- Nếu power >= 0.8 ở n=5 → 5 seeds đủ')


if __name__ == '__main__':
    main()
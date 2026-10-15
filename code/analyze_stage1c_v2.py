"""Stage 1c re-analysis (v2).

Pre-registered:
- Primary metric: NLL (unweighted)
- Secondary: Macro-F1
- Mixed model: NLL ~ log2(rank) * C(module) + (1|seed)
- Categorical check: NLL ~ C(rank) * C(module) + (1|seed)
- TOST δ_F1 = 0.01, δ_NLL = 0.02
- CI: t-based, không bootstrap
"""
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
DELTA_F1 = 0.01
DELTA_NLL = 0.02


def ci_t(x, alpha=0.05):
    """t-based CI cho mean."""
    x = np.asarray(x)
    n = len(x)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    return m, se, m - t_crit*se, m + t_crit*se


def tost_1samp(x, delta, alpha=0.05):
    """TOST cho H0: |mu| >= delta vs H1: |mu| < delta."""
    x = np.asarray(x)
    n = len(x)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha, df=n-1)
    # Lower bound test: mu > -delta
    t_low = (m - (-delta)) / se
    p_low = stats.t.sf(t_low, df=n-1)
    # Upper bound test: mu < delta
    t_high = (m - delta) / se
    p_high = stats.t.cdf(t_high, df=n-1)
    p_tost = max(p_low, p_high)
    return m, se, p_low, p_high, p_tost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    df['module'] = df['config'].apply(lambda x: x.split(':', 1)[1])

    out = {'delta_f1': DELTA_F1, 'delta_nll': DELTA_NLL}

    print('=' * 80)
    print('=== PRIMARY METRIC: NLL (unweighted) ===')
    print('=' * 80)

    # === NLL by (module, rank) ===
    print('\nMean NLL by (module, rank), 5 seeds:')
    piv_nll = df.pivot_table(index='module', columns='rank',
                             values='nll', aggfunc='mean')
    print(piv_nll.round(4).to_string())

    print('\nStd NLL by (module, rank):')
    piv_nll_std = df.pivot_table(index='module', columns='rank',
                                 values='nll', aggfunc='std')
    print(piv_nll_std.round(4).to_string())

    # === Paired delta NLL vs rank 1 ===
    print('\n' + '=' * 80)
    print('Paired ΔNLL vs rank 1 (mean ± t-CI95 across 5 seeds)')
    print('=' * 80)
    print(f'{"module":30s} | {"rank 4":>18s} | {"rank 16":>18s} | {"rank 32":>18s}')
    print('-' * 95)

    delta_nll = {}
    for mod in MODULES:
        sub = df[df.module == mod]
        piv = sub.pivot_table(index='seed', columns='rank', values='nll')
        row = [f'{mod:30s}']
        mod_d = {}
        for r in RANKS[1:]:
            d = (piv[r] - piv[1]).values
            m, se, lo, hi = ci_t(d)
            row.append(f'{m:+.4f} [{lo:+.4f},{hi:+.4f}]')
            mod_d[str(r)] = {'mean': float(m), 'ci_lo': float(lo),
                             'ci_hi': float(hi)}
        print(' | '.join(row))
        delta_nll[mod] = mod_d
    out['delta_nll'] = delta_nll

    # === TOST: equivalence test ===
    print('\n' + '=' * 80)
    print(f'TOST: H0 |ΔNLL| >= {DELTA_NLL} vs H1 |ΔNLL| < {DELTA_NLL} (rank 32 vs rank 1)')
    print('=' * 80)
    print(f'{"module":30s} | {"ΔNLL":>9s} | {"p_TOST":>9s} | {"verdict":>12s}')
    print('-' * 70)
    tost_nll = {}
    for mod in MODULES:
        sub = df[df.module == mod]
        piv = sub.pivot_table(index='seed', columns='rank', values='nll')
        d = (piv[32] - piv[1]).values
        m, se, p_low, p_high, p_tost = tost_1samp(d, DELTA_NLL)
        verdict = 'equivalent' if p_tost < 0.05 else 'not equiv'
        print(f'{mod:30s} | {m:>+9.4f} | {p_tost:>9.4f} | {verdict:>12s}')
        tost_nll[mod] = {'mean': float(m), 'p_tost': float(p_tost),
                         'equivalent': bool(p_tost < 0.05)}
    out['tost_nll'] = tost_nll

    # === Spearman rho on NLL ===
    print('\n' + '=' * 80)
    print('Spearman rho (rank vs NLL), 20 points')
    print('=' * 80)
    for mod in MODULES:
        sub = df[df.module == mod]
        rho, p = stats.spearmanr(sub['rank'], sub['nll'])
        print(f'{mod:30s} rho={rho:+.3f} p={p:.4f}')

    # === SECONDARY: Macro-F1 ===
    print('\n' + '=' * 80)
    print('=== SECONDARY METRIC: Macro-F1 ===')
    print('=' * 80)
    print(f'TOST: H0 |ΔF1| >= {DELTA_F1} vs H1 |ΔF1| < {DELTA_F1}')
    print(f'{"module":30s} | {"ΔF1":>9s} | {"p_TOST":>9s} | {"verdict":>12s}')
    print('-' * 70)
    tost_f1 = {}
    for mod in MODULES:
        sub = df[df.module == mod]
        piv = sub.pivot_table(index='seed', columns='rank', values='macro_f1')
        d = (piv[32] - piv[1]).values
        m, se, p_low, p_high, p_tost = tost_1samp(d, DELTA_F1)
        verdict = 'equivalent' if p_tost < 0.05 else 'not equiv'
        print(f'{mod:30s} | {m:>+9.4f} | {p_tost:>9.4f} | {verdict:>12s}')
        tost_f1[mod] = {'mean': float(m), 'p_tost': float(p_tost),
                        'equivalent': bool(p_tost < 0.05)}
    out['tost_f1'] = tost_f1

    # === Mixed model ===
    print('\n' + '=' * 80)
    print('=== MIXED MODEL (statsmodels) ===')
    print('=' * 80)
    try:
        import statsmodels.formula.api as smf
        df['log2rank'] = np.log2(df['rank'])
        df['module_c'] = df['module'].astype('category')

        # NLL: NLL ~ log2(rank) * C(module) + (1|seed)
        md_nll = smf.mixedlm('nll ~ log2rank * C(module)',
                              data=df, groups=df['seed'])
        res_nll = md_nll.fit(reml=True)
        print('\nMixed model on NLL:')
        print(res_nll.summary().tables[1])

        # Categorical
        md_nll_cat = smf.mixedlm('nll ~ C(rank) * C(module)',
                                  data=df, groups=df['seed'])
        res_nll_cat = md_nll_cat.fit(reml=True)
        print('\nMixed model on NLL (categorical rank):')
        # Chỉ in interaction terms
        for k, v in res_nll_cat.params.items():
            if 'C(rank)' in k and 'C(module)' in k:
                p = res_nll_cat.pvalues[k]
                print(f'  {k:60s}  coef={v:+.5f}  p={p:.4f}')

        out['mixed_nll'] = {
            'params': res_nll.params.to_dict(),
            'pvalues': res_nll.pvalues.to_dict(),
        }
    except ImportError:
        print('statsmodels not installed. Skipping mixed model.')
        print('Cài: pip install --user statsmodels')

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
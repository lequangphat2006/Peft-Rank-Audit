"""Stage 1c v3 — Holm-corrected TOST + LRT + F1 mixed model.

Pre-registered:
- Primary: NLL unweighted
- Secondary: Macro-F1
- Confirmatory tests (Holm m=6): TOST per module (rank32 - rank1)
- LRT (ML, không REML): module × rank interaction
- Report δ ∈ {0.01, 0.02, 0.03}
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


def tost_1samp(x, delta, alpha=0.05):
    x = np.asarray(x)
    n = len(x)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha, df=n-1)
    t_low = (m + delta) / se
    p_low = stats.t.sf(t_low, df=n-1)
    t_high = (m - delta) / se
    p_high = stats.t.cdf(t_high, df=n-1)
    p_tost = max(p_low, p_high)
    return m, se, p_tost


def lrt_interaction(df, metric):
    """LRT cho module × rank interaction. Dùng ML, không REML."""
    import statsmodels.formula.api as smf
    df = df.copy()
    df['log2rank'] = np.log2(df['rank'])

    # Full model
    md_full = smf.mixedlm(f'{metric} ~ log2rank * C(module)',
                          data=df, groups=df['seed'])
    res_full = md_full.fit(reml=False, method='lbfgs')

    # Null model
    md_null = smf.mixedlm(f'{metric} ~ log2rank + C(module)',
                          data=df, groups=df['seed'])
    res_null = md_null.fit(reml=False, method='lbfgs')

    lr_stat = 2 * (res_full.llf - res_null.llf)
    # df = difference in number of parameters
    df_diff = res_full.df_modelwc - res_null.df_modelwc
    p_lrt = stats.chi2.sf(lr_stat, df_diff)

    return {
        'lr_stat': float(lr_stat),
        'df': int(df_diff),
        'p_lrt': float(p_lrt),
        'full_llf': float(res_full.llf),
        'null_llf': float(res_null.llf),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    df['module'] = df['config'].apply(lambda x: x.split(':', 1)[1])
    out = {}

    # ============================================================
    # 1. TOST with Holm correction
    # ============================================================
    for metric, deltas in [('nll', [0.01, 0.02, 0.03]),
                            ('macro_f1', [0.005, 0.01, 0.02])]:
        print('=' * 80)
        print(f'TOST: {metric}, Holm-corrected (m=6)')
        print('=' * 80)
        for delta in deltas:
            print(f'\nδ = {delta}')
            print(f'{"module":30s} | {"Δ mean":>9s} | {"p_raw":>8s} | '
                  f'{"p_holm":>8s} | {"verdict":>10s}')
            print('-' * 75)

            pvals, means = [], []
            for mod in MODULES:
                sub = df[df.module == mod]
                piv = sub.pivot_table(index='seed', columns='rank',
                                      values=metric)
                d = (piv[32] - piv[1]).values
                m, se, p = tost_1samp(d, delta)
                pvals.append(p)
                means.append(m)

            adj = holm(pvals)
            for i, mod in enumerate(MODULES):
                verdict = 'equivalent' if adj[i] < 0.05 else 'not equiv'
                print(f'{mod:30s} | {means[i]:>+9.4f} | {pvals[i]:>8.4f} | '
                      f'{adj[i]:>8.4f} | {verdict:>10s}')
            out[f'tost_{metric}_d{delta}'] = {
                'means': dict(zip(MODULES, means)),
                'p_raw': dict(zip(MODULES, pvals)),
                'p_holm': dict(zip(MODULES, adj.tolist())),
            }

    # ============================================================
    # 2. LRT for module × rank interaction
    # ============================================================
    print('\n' + '=' * 80)
    print('LRT for module × rank interaction (ML, not REML)')
    print('=' * 80)
    for metric in ['nll', 'macro_f1']:
        try:
            r = lrt_interaction(df, metric)
            print(f'\n{metric}:')
            print(f'  LR stat = {r["lr_stat"]:.4f}, df = {r["df"]}, '
                  f'p = {r["p_lrt"]:.4f}')
            verdict = 'SIG' if r['p_lrt'] < 0.05 else 'n.s.'
            print(f'  Module × rank interaction: {verdict}')
            out[f'lrt_{metric}'] = r
        except Exception as e:
            print(f'  {metric}: LRT failed: {e}')

    # ============================================================
    # 3. Mixed model on F1
    # ============================================================
    print('\n' + '=' * 80)
    print('Mixed model on Macro-F1')
    print('=' * 80)
    try:
        import statsmodels.formula.api as smf
        df['log2rank'] = np.log2(df['rank'])
        md = smf.mixedlm('macro_f1 ~ log2rank * C(module)',
                         data=df, groups=df['seed'])
        res = md.fit(reml=True)
        print(res.summary().tables[1])
        out['mixed_f1'] = {
            'params': res.params.to_dict(),
            'pvalues': res.pvalues.to_dict(),
        }
    except Exception as e:
        print(f'Mixed model F1 failed: {e}')

    # ============================================================
    # 4. Seed variance check
    # ============================================================
    print('\n' + '=' * 80)
    print('Seed variance (mixed model, NLL)')
    print('=' * 80)
    try:
        import statsmodels.formula.api as smf
        md = smf.mixedlm('nll ~ log2rank * C(module)',
                         data=df, groups=df['seed'])
        res = md.fit(reml=True)
        # Group var is the seed random effect variance
        group_var = float(res.cov_re.iloc[0, 0])
        resid_var = float(res.scale)
        print(f'Seed (Group) variance: {group_var:.6f}')
        print(f'Residual variance:     {resid_var:.6f}')
        print(f'ICC (seed/total):      {group_var / (group_var + resid_var):.4f}')
        out['seed_variance'] = {
            'group_var': group_var,
            'resid_var': resid_var,
            'icc': group_var / (group_var + resid_var) if (group_var + resid_var) > 0 else 0.0,
        }
    except Exception as e:
        print(f'Failed: {e}')

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
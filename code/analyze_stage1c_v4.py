"""Stage 1c v4 — fixed-effects seed, F-test cho interaction, pooled slope.

Sửa so với v3:
- Fixed effect seed: C(seed) trong OLS, không dùng mixed model
- F-test thay LRT (tránh nan)
- Assert finiteness trước khi in kết luận
- Báo cáo TOST ở nhiều δ
- Pooled slope cho F1 và NLL với 4 chữ số
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse, json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
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


def interaction_tests(df, y):
    """F-test cho interaction dùng fixed-effect seed. Không dùng mixed model."""
    d = df.copy()
    d['lr'] = np.log2(d['rank'])
    out = {}

    # 1. Slope heterogeneity: C(module)*lr vs C(module) + lr
    full = smf.ols(f'{y} ~ C(seed) + C(module)*lr', d).fit()
    red = smf.ols(f'{y} ~ C(seed) + C(module) + lr', d).fit()
    a = sm.stats.anova_lm(red, full)
    out['slope_het'] = {
        'F': float(a['F'][1]),
        'df_diff': int(a['df_diff'][1]),
        'p': float(a['Pr(>F)'][1]),
    }

    # 2. Categorical interaction: C(module)*C(rank)
    full = smf.ols(f'{y} ~ C(seed) + C(module)*C(rank)', d).fit()
    red = smf.ols(f'{y} ~ C(seed) + C(module) + C(rank)', d).fit()
    a = sm.stats.anova_lm(red, full)
    out['cat_inter'] = {
        'F': float(a['F'][1]),
        'df_diff': int(a['df_diff'][1]),
        'p': float(a['Pr(>F)'][1]),
    }

    # 3. Pooled slope (common slope across modules, main effect of log2 rank)
    m = smf.ols(f'{y} ~ C(seed) + C(module) + lr', d).fit()
    ci = m.conf_int().loc['lr']
    out['pooled_slope'] = {
        'coef': float(m.params['lr']),
        'ci_lo': float(ci[0]),
        'ci_hi': float(ci[1]),
        'p': float(m.pvalues['lr']),
        'se': float(m.bse['lr']),
    }

    # Verify finiteness
    for k, v in out.items():
        for kk, vv in v.items():
            if isinstance(vv, float):
                assert np.isfinite(vv), f'{k}.{kk} not finite = {vv}'

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    df['module'] = df['config'].apply(lambda x: x.split(':', 1)[1])
    df['rank'] = df['rank'].astype(int)
    df['seed'] = df['seed'].astype(int)

    out = {}

    # ============================================================
    # 1. F-tests (interaction)
    # ============================================================
    print('=' * 80)
    print('F-tests (fixed-effect seed)')
    print('=' * 80)

    for metric in ['nll', 'macro_f1']:
        print(f'\n>>> {metric}')
        try:
            r = interaction_tests(df, metric)
            print(f'  Slope heterogeneity (module × log2 rank), df={r["slope_het"]["df_diff"]}:')
            print(f'    F = {r["slope_het"]["F"]:.4f}, p = {r["slope_het"]["p"]:.4f}')
            verdict = 'SIG' if r['slope_het']['p'] < 0.05 else 'n.s.'
            print(f'    → {verdict}')
            print(f'  Categorical interaction (module × rank), df={r["cat_inter"]["df_diff"]}:')
            print(f'    F = {r["cat_inter"]["F"]:.4f}, p = {r["cat_inter"]["p"]:.4f}')
            verdict = 'SIG' if r['cat_inter']['p'] < 0.05 else 'n.s.'
            print(f'    → {verdict}')
            print(f'  Pooled slope (common across modules):')
            ps = r['pooled_slope']
            print(f'    coef = {ps["coef"]:+.6f}  '
                  f'CI95 = [{ps["ci_lo"]:+.6f}, {ps["ci_hi"]:+.6f}]  '
                  f'se = {ps["se"]:.6f}  p = {ps["p"]:.4f}')
            out[f'ftest_{metric}'] = r
        except AssertionError as e:
            print(f'  ERROR: {e}')
            print(f'  → NOT reporting verdict, xem lại data/design')
        except Exception as e:
            print(f'  Failed: {e}')

    # ============================================================
    # 2. TOST at multiple δ with Holm
    # ============================================================
    print('\n' + '=' * 80)
    print('TOST at multiple δ, Holm-corrected (m=6)')
    print('=' * 80)

    tost_out = {}
    for metric, deltas in [('nll', [0.01, 0.02, 0.03]),
                            ('macro_f1', [0.005, 0.01, 0.02])]:
        for delta in deltas:
            print(f'\n>>> {metric}, δ = {delta}')
            print(f'  {"module":30s} | {"Δ mean":>10s} | {"p_raw":>8s} | '
                  f'{"p_holm":>8s} | {"verdict":>10s}')
            print('  ' + '-' * 78)

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
                print(f'  {mod:30s} | {means[i]:>+10.5f} | {pvals[i]:>8.4f} | '
                      f'{adj[i]:>8.4f} | {verdict:>10s}')
            tost_out[f'{metric}_d{delta}'] = {
                'delta': delta,
                'means': dict(zip(MODULES, means)),
                'p_raw': dict(zip(MODULES, pvals)),
                'p_holm': dict(zip(MODULES, adj.tolist())),
            }
    out['tost'] = tost_out

    # ============================================================
    # 3. Seed variance via fixed-effect ANOVA
    # ============================================================
    print('\n' + '=' * 80)
    print('Seed effect: variance explained by C(seed)')
    print('=' * 80)
    for metric in ['nll', 'macro_f1']:
        full = smf.ols(f'{metric} ~ C(seed) + C(module) + C(rank)',
                       df).fit()
        red = smf.ols(f'{metric} ~ C(module) + C(rank)', df).fit()
        a = sm.stats.anova_lm(red, full)
        # eta^2 = SS_seed / SS_total
        ss_seed = a['ss_diff'][1] if 'ss_diff' in a.columns else a['sum_sq'][0]
        ss_total = full.centered_tss
        eta2 = float(ss_seed / ss_total)
        print(f'{metric}: SS_seed={ss_seed:.6f}, SS_total={ss_total:.6f}, '
              f'η²_seed={eta2:.4f}')
        out[f'seed_eta2_{metric}'] = eta2

    # ============================================================
    # 4. Summary
    # ============================================================
    print('\n' + '=' * 80)
    print('SUMMARY (chặt nhất có thể nói)')
    print('=' * 80)
    f1_ps = out.get('ftest_macro_f1', {}).get('pooled_slope', {})
    if f1_ps:
        coef = f1_ps['coef']
        lo = f1_ps['ci_lo']
        hi = f1_ps['ci_hi']
        # Total effect from rank 1 to rank 32 = coef * log2(32/1) = coef * 5
        total_coef = coef * 5
        total_lo = lo * 5
        total_hi = hi * 5
        print(f'F1 pooled slope per log2(rank): {coef:+.6f} '
              f'[{lo:+.6f}, {hi:+.6f}]')
        print(f'  → Total effect rank 1→32: {total_coef:+.5f} '
              f'[{total_lo:+.5f}, {total_hi:+.5f}]')

    nll_ps = out.get('ftest_nll', {}).get('pooled_slope', {})
    if nll_ps:
        coef = nll_ps['coef']
        lo = nll_ps['ci_lo']
        hi = nll_ps['ci_hi']
        total_coef = coef * 5
        total_lo = lo * 5
        total_hi = hi * 5
        print(f'NLL pooled slope per log2(rank): {coef:+.6f} '
              f'[{lo:+.6f}, {hi:+.6f}]')
        print(f'  → Total effect rank 1→32: {total_coef:+.5f} '
              f'[{total_lo:+.5f}, {total_hi:+.5f}]')

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
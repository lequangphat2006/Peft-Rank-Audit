import pandas as pd
import numpy as np
import os

# Hàm tính hiệu ứng rank 1->32
def calc_rank_effect(df):
    df['log2_rank'] = np.log2(df['rank'])
    effects = []
    for seed in df['seed'].unique():
        d = df[df['seed'] == seed]
        if len(d) < 2: continue
        z = np.polyfit(d['log2_rank'], d['macro_f1'], 1)
        effects.append(z[0] * 5) # Nhân với log2(32)
    return np.mean(effects), np.std(effects, ddof=1)

# Lấy dữ liệu B1X (Tune LR theo rank)
pho_main = pd.read_csv("results/ext/phobert/B1X_total.csv")
xlm_main = pd.read_csv("results/ext/xlmr/XLM_total_v2.csv")

# Lấy dữ liệu Fixed LR 2e-4
pho_fixed = pho_main[(pho_main['lr'] == 0.0002) & (pho_main['cell'] == 'B1X')]
xlm_fixed = xlm_main[xlm_main['lr'] == 0.0002]

specs = []
for name, pho_data, xlm_data in [
    ("Tuned LR", pho_main, xlm_main),
    ("Fixed LR (2e-4)", pho_fixed, xlm_fixed)
]:
    p_mean, p_sd = calc_rank_effect(pho_data)
    x_mean, x_sd = calc_rank_effect(xlm_data)
    specs.append({"Protocol": name, "PhoBERT": p_mean, "XLM-R": x_mean})

print("=== SPECIFICATION CURVE: RANK EFFECT (1->32) ===")
print(pd.DataFrame(specs).to_string(index=False))
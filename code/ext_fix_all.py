# code/ext_fix_all.py - Sửa lỗi triệt để
import pandas as pd
import numpy as np
from scipy import stats
import json
import os

def get_best_lr(lr_map, rank):
    # Hàm tự động xử lý mọi cấu trúc file JSON
    val = lr_map.get(str(rank), None)
    if val is None:
        val = lr_map.get(rank, None)
    
    if isinstance(val, dict):
        return val.get("best_lr", -1)
    return val if val is not None else -1

def calc_effect(csv_file, lr_map_file):
    df = pd.read_csv(csv_file)
    with open(lr_map_file, 'r') as f:
        raw_map = json.load(f)
    
    df['log2_rank'] = np.log2(df['rank'])
    
    # Lọc dữ liệu dùng vòng lặp an toàn
    mask = []
    for idx, row in df.iterrows():
        best_lr = get_best_lr(raw_map, row['rank'])
        mask.append(np.isclose(row['lr'], best_lr))
    
    df = df[mask]
    
    effects = []
    for seed in df['seed'].unique():
        df_seed = df[df['seed'] == seed]
        if len(df_seed['rank'].unique()) < 2: continue
        slope, _, _, _, _ = stats.linregress(df_seed['log2_rank'], df_seed['macro_f1'])
        effects.append(slope * 5)
    return np.mean(effects), np.std(effects, ddof=1), len(effects)

print("=== SPEC CURVE (Đã lọc nhánh) ===")
mean_p, sd_p, n_p = calc_effect("results/ext/t2/T2_PHO.csv", "results/ext/phobert/lr_map_best.json")
mean_x, sd_x, n_x = calc_effect("results/ext/t2/T2_XLM.csv", "results/ext/xlmr/lr_map_best.json")
print(f"PhoBERT: {mean_p:+.4f} (SD: {sd_p:.4f}, N={n_p})")
print(f"XLM-R:   {mean_x:+.4f} (SD: {sd_x:.4f}, N={n_x})")
print(f"Delta (P - X): {mean_p - mean_x:+.4f}")

# 2. SỬA LỖI META-ANALYSIS (Dùng t-distribution, Hartung-Knapp)
print("\n=== META-ANALYSIS (Sửa lỗi z-score) ===")
# Dữ liệu từ 4 lần lặp (B1X, T2, T3-F2, T3-F3)
deltas = np.array([0.0073, 0.0014, -0.0104, 0.0072])
ses = np.array([0.0032, 0.0037, 0.0051, 0.0045])
k = len(deltas)

# Hartung-Knapp
w_fixed = 1 / ses**2
mean_fixed = np.sum(deltas * w_fixed) / np.sum(w_fixed)
Q = np.sum(w_fixed * (deltas - mean_fixed)**2)
C = np.sum(w_fixed) - np.sum(w_fixed**2) / np.sum(w_fixed)
tau2 = max(0, (Q - k + 1) / C)
tau = np.sqrt(tau2)

w_re = 1 / (ses**2 + tau2)
mean_re = np.sum(deltas * w_re) / np.sum(w_re)
se_re = np.sqrt(1 / np.sum(w_re))

# Dùng t với df = k-1
t_crit = stats.t.ppf(0.975, df=k-1)
ci_lower = mean_re - t_crit * se_re
ci_upper = mean_re + t_crit * se_re

print(f"Pooled Delta: {mean_re:+.4f}")
print(f"Tau: {tau:.4f}")
print(f"95% CI (t, df=3): [{ci_lower:+.4f}, {ci_upper:+.4f}]")
print("Lưu ý: XLM-R ở B1X và T2 là cùng dữ liệu, nên số lần lặp độc lập thực tế < 4.")
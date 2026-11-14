import sys
import os
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_common import read_csv, load_lr_map, select_branch, per_seed_effects

def get_effect(csv_file, lr_map_file):
    try:
        df = read_csv(csv_file)
        df = df[df['epochs'] == 6] # Chỉ lấy 6 epochs
        # Lọc nhánh LR (giống hệt ext_confirmatory)
        df_branch = select_branch(df, load_lr_map(lr_map_file))
        # Tính hiệu ứng theo từng seed
        sp, _ = per_seed_effects(df_branch, 'macro_f1', ranks=[1, 4, 8, 16, 32])
        return sp['total'].mean(), sp['total'].std(ddof=1), len(sp)
    except Exception as e:
        print(f"Lỗi ở file {csv_file}: {e}")
        return None, None, 0

datasets = {
    "B1X (Main)": {"P": "results/ext/phobert/B1X_total.csv", "P_MAP": "results/ext/phobert/lr_map_best.json", 
                   "X": "results/ext/xlmr/XLM_total_v2.csv", "X_MAP": "results/ext/xlmr/lr_map_best.json"},
    "T2 (Session-Blocked)": {"P": "results/ext/t2/T2_PHO.csv", "P_MAP": "results/ext/phobert/lr_map_best.json", 
                             "X": "results/ext/t2/T2_XLM.csv", "X_MAP": "results/ext/xlmr/lr_map_best.json"},
    "T3-F2 (New Split)": {"P": "results/ext/t3/T3_PHO_F2.csv", "P_MAP": "results/ext/phobert/lr_map_best.json", 
                          "X": "results/ext/t3/T3_XLM_F2.csv", "X_MAP": "results/ext/xlmr/lr_map_best.json"},
    "T3-F3 (New Split)": {"P": "results/ext/t3/T3_PHO_F3.csv", "P_MAP": "results/ext/phobert/lr_map_best.json", 
                          "X": "results/ext/t3/T3_XLM_F3.csv", "X_MAP": "results/ext/xlmr/lr_map_best.json"}
}

print("=== META-ANALYSIS OF MODEL x RANK INTERACTION ===")
results = []
for name, paths in datasets.items():
    mean_p, sd_p, n_p = get_effect(paths["P"], paths["P_MAP"])
    mean_x, sd_x, n_x = get_effect(paths["X"], paths["X_MAP"])
    
    if mean_p is not None and mean_x is not None:
        diff = mean_p - mean_x
        se_diff = np.sqrt((sd_p**2 / n_p) + (sd_x**2 / n_x))
        results.append({"Iteration": name, "Delta_E1": diff, "SE": se_diff, "N": n_p})
        print(f"{name:20s} | Delta E1: {diff:+.4f} (SE: {se_diff:.4f}, N={n_p})")

if results:
    deltas = np.array([r["Delta_E1"] for r in results])
    ses = np.array([r["SE"] for r in results])
    
    weights_fixed = 1 / ses**2
    mean_fixed = np.sum(deltas * weights_fixed) / np.sum(weights_fixed)
    Q = np.sum(weights_fixed * (deltas - mean_fixed)**2)
    C = np.sum(weights_fixed) - np.sum(weights_fixed**2) / np.sum(weights_fixed)
    tau2 = max(0, (Q - len(deltas) + 1) / C)
    tau = np.sqrt(tau2)
    
    weights_re = 1 / (ses**2 + tau2)
    mean_re = np.sum(deltas * weights_re) / np.sum(weights_re)
    se_re = np.sqrt(1 / np.sum(weights_re))
    
    print("\n--- RANDOM EFFECTS MODEL ---")
    print(f"Pooled Delta E1: {mean_re:+.4f}")
    print(f"Tau (Between-iteration SD): {tau:.4f}")
    print(f"95% CI: [{mean_re - 1.96*se_re:+.4f}, {mean_re + 1.96*se_re:+.4f}]")
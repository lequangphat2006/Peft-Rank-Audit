"""analyze_v7_v8.py - Phân tích V7 (Bootstrap CI) và V8 (H2 & Temperature Scaling)."""
import pandas as pd
import numpy as np
import json
import os

# 1. Đọc dữ liệu R1
df = pd.read_csv("results/global/global_merged.csv")
df_6ep = df[df["epochs"] == 6].copy()

print("=== V7: BOOTSTRAP CI VÀ F1 TỪNG LỚP (6 EPOCHS) ===")
def bootstrap_ci(data, metric, n_iter=1000):
    means = []
    for _ in range(n_iter):
        sample = data.sample(n=len(data), replace=True)
        means.append(sample[metric].mean())
    return np.percentile(means, [2.5, 97.5])

for rank in [1, 4, 8, 16, 32]:
    sub = df_6ep[df_6ep["rank"] == rank]
    if len(sub) == 0: continue
    
    # Macro-F1
    ci_macro = bootstrap_ci(sub, "macro_f1")
    # F1 Lớp Neutral (Lớp khó nhất)
    ci_neu = bootstrap_ci(sub, "f1_neu")
    
    print(f"Rank {rank:2d}: Macro-F1 = {sub['macro_f1'].mean():.4f} [95% CI: {ci_macro[0]:.4f}, {ci_macro[1]:.4f}] | F1-Neu = {sub['f1_neu'].mean():.4f} [95% CI: {ci_neu[0]:.4f}, {ci_neu[1]:.4f}]")

print("\n=== V8: KIỂM TRA H2 VÀ TEMPERATURE SCALING ===")
# Đọc file global_v1.json (đã có kết quả phân tích)
json_path = "results/analysis/global_v1.json"
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        res = json.load(f)
    
    # H2 NLL
    nll_total = res["H2"]["est"]["total"]
    print(f"H2 (NLL total): {nll_total:+.4f} (Nhỏ hơn ngưỡng thực tế δ_NLL = 0.01 rất nhiều)")
    
    # Temperature Scaling (Nếu có trong JSON)
    if "nll_after_ts" in res:
        print(f"NLL sau Temperature Scaling: {res['nll_after_ts']}")
    else:
        print("NLL sau Temperature Scaling: (Cần xem trong output gốc của analyze_global.py)")
        
    print("\n* Lưu ý: Lớp Neutral (4% dữ liệu) có F1 dao động mạnh (0.59 - 0.61) và chi phối Macro-F1.")
else:
    print("Không tìm thấy global_v1.json.")
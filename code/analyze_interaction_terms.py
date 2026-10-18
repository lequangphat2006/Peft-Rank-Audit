"""analyze_interaction_terms.py - Tính độ dốc rank ở 3 vs 6 epoch cho cả 2 nhánh LR."""
import pandas as pd
import numpy as np
import json
import os

os.makedirs("results/analysis", exist_ok=True)

def calc_slope(df, epochs, label):
    subset = df[df["epochs"] == epochs]
    # Tính mean F1 theo log2(rank) để lấy slope
    means = subset.groupby("rank")["macro_f1"].mean()
    x = np.log2(means.index)
    y = means.values
    slope, _ = np.polyfit(x, y, 1)
    total_effect = slope * np.log2(32/1)
    return total_effect

# 1. Nhánh LR per rank (R1)
df_r1 = pd.read_csv("results/global/global_merged.csv")
r1_3ep = calc_slope(df_r1, 3, "R1 3ep")
r1_6ep = calc_slope(df_r1, 6, "R1 6ep")
r1_interaction = r1_6ep - r1_3ep

# 2. Nhánh LR cố định (V1 và V1b)
df_v1 = pd.read_csv("results/v1_fixed_lr/V1_merged.csv")
df_v1["epochs"] = 3
df_v1b = pd.read_csv("results/posthoc/V1b_merged.csv")
df_v1b["epochs"] = 6
df_fixed = pd.concat([df_v1, df_v1b], ignore_index=True)

fixed_3ep = calc_slope(df_fixed, 3, "Fixed 3ep")
fixed_6ep = calc_slope(df_fixed, 6, "Fixed 6ep")
fixed_interaction = fixed_6ep - fixed_3ep

out = {
    "R1_LR_per_rank": {"effect_3ep": round(r1_3ep, 4), "effect_6ep": round(r1_6ep, 4), "interaction": round(r1_interaction, 4)},
    "Fixed_LR_2e4": {"effect_3ep": round(fixed_3ep, 4), "effect_6ep": round(fixed_6ep, 4), "interaction": round(fixed_interaction, 4)}
}

with open("results/analysis/interaction_terms.json", "w") as f:
    json.dump(out, f, indent=2)

print("✅ Đã tính toán xong Interaction Terms:")
print(json.dumps(out, indent=2))
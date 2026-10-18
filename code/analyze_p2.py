"""analyze_p2.py - Vẽ Heatmap và tính Kendall Tau cho AdaLoRA."""
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau

os.makedirs("results/analysis/plots", exist_ok=True)

# ==========================================
# 1. VẼ HEATMAP RANK x LR (Từ dữ liệu Tuning)
# ==========================================
print("=== 1. Vẽ Heatmap Rank x LR ===")
df_tune = pd.read_csv("results/tuning/tuning_full.csv")
pivot = df_tune.pivot_table(index="rank", columns="lr", values="macro_f1", aggfunc="mean")

plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Macro-F1'})
plt.title("Macro-F1 (Calibration) by Rank and LR\n(Darker = Better, Light/White = Collapse)")
plt.xlabel("Learning Rate")
plt.ylabel("LoRA Rank")
plt.tight_layout()
plt.savefig("results/analysis/plots/heatmap_rank_lr.png", dpi=300)
print("Đã lưu: results/analysis/plots/heatmap_rank_lr.png")

# ==========================================
# 2. TÍNH KENDALL TAU CHO ADALORA RANK
# ==========================================
print("\n=== 2. Kiểm tra độ ổn định rank của AdaLoRA (Kendall Tau) ===")
rank_dir = "results/adalora/adalora_rank"
if os.path.exists(rank_dir):
    files = [f for f in os.listdir(rank_dir) if f.endswith(".json")]
    # Đọc 3 file seed 200, 201, 202
    ranks_by_seed = {}
    for f in files:
        with open(os.path.join(rank_dir, f), "r") as file:
            data = json.load(file)
            seed = data["run_id"].split("__")[1] # Lấy 's200'
            if seed in ["s200", "s201", "s202"]:
                ranks_by_seed[seed] = data["ranks"]
    
    if len(ranks_by_seed) == 3:
        modules = list(ranks_by_seed["s200"].keys())
        r200 = [ranks_by_seed["s200"][m] for m in modules]
        r201 = [ranks_by_seed["s201"][m] for m in modules]
        r202 = [ranks_by_seed["s202"][m] for m in modules]
        
        tau_200_201, _ = kendalltau(r200, r201)
        tau_200_202, _ = kendalltau(r200, r202)
        
        print(f"Kendall Tau (Seed 200 vs 201): {tau_200_201:.4f}")
        print(f"Kendall Tau (Seed 200 vs 202): {tau_200_202:.4f}")
        if tau_200_201 < 0.2:
            print("=> Kết luận: Phân bổ rank của AdaLoRA KHÔNG ổn định giữa các seed (dao động mạnh).")
        else:
            print("=> Kết luận: Phân bổ rank của AdaLoRA có xu hướng ổn định.")
    else:
        print("Không đủ 3 file rank của seed 200-202.")
else:
    print("Không tìm thấy thư mục adalora_rank.")
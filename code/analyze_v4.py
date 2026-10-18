"""analyze_v4.py - Phân tích kết quả Tune LR cho PhoBERT."""
import pandas as pd
import json
import os

print("=== 1. KẾT QUẢ TUNE LR (CALIBRATION) ===")
dfs = [pd.read_csv(f"results/v4_phobert/V4_tune_gpu{g}.csv") for g in (0, 1)]
tune_df = pd.concat(dfs, ignore_index=True)

# Tính F1 trung bình theo rank và lr
summary = tune_df.groupby(["rank", "lr"])["macro_f1"].mean().reset_index()
for rank in [1, 4, 16, 32]:
    sub = summary[summary["rank"] == rank].sort_values("macro_f1", ascending=False)
    best = sub.iloc[0]
    print(f"Rank {rank:2d}: Best LR = {best['lr']:.0e} (F1_cal = {best['macro_f1']:.4f})")

print("\n=== 2. KẾT QUẢ B1 FINAL (TEST, 8 SEEDS) ===")
if os.path.exists("results/v4_phobert/B1_final_merged.csv"):
    final_df = pd.read_csv("results/v4_phobert/B1_final_merged.csv")
    
    # Tính F1 và NLL trung bình theo rank
    final_summary = final_df.groupby("rank").agg(
        f1_mean=("macro_f1", "mean"),
        f1_std=("macro_f1", "std"),
        nll_mean=("nll", "mean"),
        f1_neu_mean=("f1_neu", "mean")
    ).round(4)
    
    print(final_summary.to_string())
else:
    print("Không tìm thấy B1_final_merged.csv")
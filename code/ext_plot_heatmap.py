import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("results/ext/figures", exist_ok=True)

for model, csv_file in [("PhoBERT", "results/ext/phobert/tune_all.csv"), ("XLM-R", "results/tuning/tuning_full.csv")]:
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        pivot = df.pivot_table(index='lr', columns='rank', values='macro_f1', aggfunc='mean')
        
        plt.figure(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f")
        plt.title(f"F1 Heatmap: {model} (Rank × LR)")
        plt.xlabel("LoRA Rank")
        plt.ylabel("Learning Rate")
        plt.gca().invert_yaxis() # LR thấp lên trên
        plt.tight_layout()
        plt.savefig(f"results/ext/figures/heatmap_{model}.png", dpi=300)
        plt.close()
        print(f"Đã lưu heatmap cho {model}!")
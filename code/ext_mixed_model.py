import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

# Load dữ liệu 4 lần lặp
files = {
    "B1X": ("results/ext/phobert/B1X_total.csv", "results/ext/xlmr/XLM_total_v2.csv"),
    "T2": ("results/ext/t2/T2_PHO.csv", "results/ext/t2/T2_XLM.csv"),
    "T3-F2": ("results/ext/t3/T3_PHO_F2.csv", "results/ext/t3/T3_XLM_F2.csv"),
    "T3-F3": ("results/ext/t3/T3_PHO_F3.csv", "results/ext/t3/T3_XLM_F3.csv")
}

dfs = []
for iteration, (p_file, x_file) in files.items():
    if os.path.exists(p_file) and os.path.exists(x_file):
        dp = pd.read_csv(p_file); dp['model'] = 'PhoBERT'; dp['iteration'] = iteration
        dx = pd.read_csv(x_file); dx['model'] = 'XLM-R'; dx['iteration'] = iteration
        dfs.append(pd.concat([dp, dx]))

df_all = pd.concat(dfs)
df_all = df_all[df_all['epochs'] == 6]
df_all['log2_rank'] = np.log2(df_all['rank'])
df_all['uid'] = df_all['iteration'] + df_all['model'] + df_all['seed'].astype(str)

# Mixed model: F1 ~ log2(rank) * model + (1 | iteration) + (1 | iteration:seed)
md = smf.mixedlm("macro_f1 ~ log2_rank * C(model)", df_all, 
                 groups=df_all["iteration"], 
                 vc_formula={"seed": "0 + C(seed)"}).fit(reml=True)

print("=== MIXED-EFFECTS MODEL (F1 ~ log2(rank) * Model) ===")
print(md.summary().tables[1])
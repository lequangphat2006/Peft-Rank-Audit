import pandas as pd
import numpy as np
import json
import scipy.stats as stats

# ==========================================
# 1. TOST VÀ CI CHO B1 (PHOBERT)
# ==========================================
print("=== TOST CHO PHOBERT (B1 Final) ===")
df = pd.read_csv("results/v4_phobert/B1_final_merged.csv")

# Lấy dữ liệu của rank 1 và rank 32, ghép cặp theo seed
pivot = df.pivot(index="seed", columns="rank", values="macro_f1")
diff = pivot[32] - pivot[1] # Sự khác biệt (Rank 32 - Rank 1)

mean_diff = diff.mean()
se_diff = diff.std(ddof=1) / np.sqrt(len(diff))
df_val = len(diff) - 1
t_crit = stats.t.ppf(0.95, df_val) # 90% CI

ci_lower = mean_diff - t_crit * se_diff
ci_upper = mean_diff + t_crit * se_diff

print(f"Sự khác biệt (Rank 32 - Rank 1): {mean_diff:+.4f}")
print(f"90% CI: [{ci_lower:+.4f}, {ci_upper:+.4f}]")
print(f"Ngưỡng tương đương (δ): 0.01")

# TOST (hai kiểm định một phía)
t_lower = (mean_diff - (-0.01)) / se_diff
t_upper = (mean_diff - 0.01) / se_diff
p_lower = 1 - stats.t.cdf(t_lower, df_val)
p_upper = stats.t.cdf(t_upper, df_val)
p_tost = max(p_lower, p_upper)

if p_tost < 0.05:
    print("Kết luận TOST: TƯƠNG ĐƯƠNG (nằm trong ±0.01)")
else:
    print("Kết luận TOST: KHÔNG tương đương (hiệu ứng > 0.01)")

# ==========================================
# 2. BOOTSTRAP CHUẨN TRÊN MẪU TEST (V7)
# ==========================================
print("\n=== BOOTSTRAP TRÊN TEST (Chờ Logits) ===")
print("Lưu ý: Script này cần file logits (.npz) từ Kaggle. Sẽ chạy sau khi tải logits về.")
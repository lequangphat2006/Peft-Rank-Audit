"""analyze_v7_true.py - Bootstrap CI trên mẫu test và loại bỏ near-duplicate."""
import os, glob, json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LOGITS_DIR = "results/global/logits"
RANKS = [1, 4, 8, 16, 32]

# 1. Lấy nhãn thực từ file npz đầu tiên
npz_files = glob.glob(f"{LOGITS_DIR}/*.npz")
if not npz_files:
    print("Không tìm thấy file npz.")
    exit()

data = np.load(npz_files[0])
y_true = data["eval_gold"]
n_samples = len(y_true)

# 2. Load toàn bộ logits vào RAM (chỉ lấy epoch 6)
logits_data = {}
for r in RANKS:
    logits_data[r] = []
    files = glob.glob(f"{LOGITS_DIR}/R1__s*_global__L-1__r{r}__e6*.npz")
    for f in files:
        d = np.load(f)
        logits_data[r].append(d["eval_logits"])

# 3. Hàm tính F1
def calc_f1(logits, y_true, indices=None):
    if indices is not None:
        logits = logits[indices]
        y_true_sub = y_true[indices]
    else:
        y_true_sub = y_true
    preds = np.argmax(logits, axis=1)
    return f1_score(y_true_sub, preds, average="macro"), f1_score(y_true_sub, preds, average=None, labels=[0,1,2])[1]

# 4. Bootstrap trên mẫu test (Resample indices của 1599 mẫu)
print("=== KẾT QUẢ BOOTSTRAP 95% CI (RESAMPLE TEST SAMPLES) ===")
np.random.seed(42)
n_boot = 1000
indices = np.arange(n_samples)

for r in RANKS:
    if not logits_data[r]: continue
    
    # Tính điểm trung bình gốc
    base_macros = []
    base_neus = []
    for logit in logits_data[r]:
        m, n = calc_f1(logit, y_true)
        base_macros.append(m)
        base_neus.append(n)
    mean_macro = np.mean(base_macros)
    mean_neu = np.mean(base_neus)
    
    # Bootstrap
    boot_macro = []
    boot_neu = []
    for _ in range(n_boot):
        idx = np.random.choice(indices, size=n_samples, replace=True)
        seed_macros = []
        seed_neus = []
        for logit in logits_data[r]:
            m, n = calc_f1(logit, y_true, indices=idx)
            seed_macros.append(m)
            seed_neus.append(n)
        boot_macro.append(np.mean(seed_macros))
        boot_neu.append(np.mean(seed_neus))
        
    ci_macro = np.percentile(boot_macro, [2.5, 97.5])
    ci_neu = np.percentile(boot_neu, [2.5, 97.5])
    
    print(f"Rank {r:2d}: Macro-F1 = {mean_macro:.4f} [95% CI: {ci_macro[0]:.4f}, {ci_macro[1]:.4f}] | F1-Neu = {mean_neu:.4f} [95% CI: {ci_neu[0]:.4f}, {ci_neu[1]:.4f}]")

# 5. Loại bỏ near-duplicate
print("\n=== KIỂM TRA LOẠI BỎ NEAR-DUPLICATE ===")
nd_path = "results/analysis/near_dup_vsfc.json"
if os.path.exists(nd_path):
    with open(nd_path, "r") as f:
        nd_data = json.load(f)
    print("Đã tìm thấy file near_dup_vsfc.json.")
    # In ra cấu trúc để kiểm tra
    if isinstance(nd_data, dict):
        print("Keys:", list(nd_data.keys()))
        if "test_indices" in nd_data:
            nd_indices = set(nd_data["test_indices"])
            clean_indices = [i for i in range(n_samples) if i not in nd_indices]
            print(f"Đã loại bỏ {len(nd_indices)} mẫu trùng. Còn lại {len(clean_indices)} mẫu test.")
            
            # Tính lại F1 trên tập sạch
            for r in RANKS:
                if not logits_data[r]: continue
                clean_macros = []
                clean_neus = []
                for logit in logits_data[r]:
                    m, n = calc_f1(logit, y_true, indices=clean_indices)
                    clean_macros.append(m)
                    clean_neus.append(n)
                print(f"Rank {r:2d} (Clean): Macro-F1 = {np.mean(clean_macros):.4f} | F1-Neu = {np.mean(clean_neus):.4f}")
    else:
        print("Cấu trúc JSON không phải dict. Kiểm tra lại file.")
else:
    print("Không tìm thấy file near_dup_vsfc.json. Bỏ qua.")
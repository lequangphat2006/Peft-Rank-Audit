"""analyze_v7_clean.py - Tính lại F1 sau khi loại bỏ near-duplicate."""
import os, glob, json
import numpy as np
from sklearn.metrics import f1_score

LOGITS_DIR = "results/global/logits"
RANKS = [1, 4, 8, 16, 32]

# Load data
npz_files = glob.glob(f"{LOGITS_DIR}/*.npz")
data = np.load(npz_files[0])
y_true = data["eval_gold"]
n_samples = len(y_true)

logits_data = {r: [] for r in RANKS}
for r in RANKS:
    files = glob.glob(f"{LOGITS_DIR}/R1__s*_global__L-1__r{r}__e6*.npz")
    for f in files:
        logits_data[r].append(np.load(f)["eval_logits"])

# Đọc cấu trúc near_dup
nd_path = "results/analysis/near_dup_vsfc.json"
with open(nd_path, "r") as f:
    nd_data = json.load(f)

# Bóc tách mảng indices của ngưỡng 0.95 (tùy thuộc cấu trúc file, ta thử các khả năng)
test_block = nd_data.get("test", {})
print("Cấu trúc bên trong key 'test':", type(test_block).__name__)

# Giả sử file lưu theo dạng: {"0.95": {"indices": [...]}} hoặc {"0.95": [...]}
nd_indices = []
if isinstance(test_block, dict):
    # Tìm key chứa '0.95' (có thể là '0.95' hoặc '0.950')
    target_key = None
    for k in test_block.keys():
        if "0.95" in k:
            target_key = k
            break
    
    if target_key:
        val = test_block[target_key]
        if isinstance(val, dict) and "indices" in val:
            nd_indices = val["indices"]
        elif isinstance(val, list):
            nd_indices = val
        print(f"Tìm thấy ngưỡng {target_key}, có {len(nd_indices)} mẫu trùng.")
elif isinstance(test_block, list):
    nd_indices = test_block
    print(f"Key 'test' là một mảng, có {len(nd_indices)} mẫu trùng.")

# Nếu không tìm thấy, in ra chi tiết để debug
if not nd_indices:
    print("Không trích xuất được danh sách indices. In chi tiết cấu trúc:")
    print(json.dumps(test_block, indent=2)[:500])
    exit()

# Tính lại F1 trên tập sạch
nd_set = set(nd_indices)
clean_indices = [i for i in range(n_samples) if i not in nd_set]
print(f"\nĐã loại bỏ {len(nd_set)} mẫu test bị trùng. Còn lại {len(clean_indices)} mẫu.")

def calc_f1(logit, y_true, indices):
    preds = np.argmax(logit[indices], axis=1)
    y_sub = y_true[indices]
    macro = f1_score(y_sub, preds, average="macro")
    neu = f1_score(y_sub, preds, average=None, labels=[0,1,2])[1]
    return macro, neu

print("\n=== KẾT QUẢ TRÊN TẬP TEST ĐÃ LOẠI BỎ NEAR-DUPLICATE ===")
for r in RANKS:
    if not logits_data[r]: continue
    macs, neus = [], []
    for logit in logits_data[r]:
        m, n = calc_f1(logit, y_true, clean_indices)
        macs.append(m)
        neus.append(n)
    print(f"Rank {r:2d}: Macro-F1 = {np.mean(macs):.4f} | F1-Neu = {np.mean(neus):.4f}")
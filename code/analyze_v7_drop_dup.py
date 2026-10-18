"""analyze_v7_drop_dup.py - Tính toán lại near-dup và lọc trực tiếp."""
import os, glob
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LOGITS_DIR = "results/global/logits"
RANKS = [1, 4, 8, 16, 32]

# 1. Load dữ liệu text
train_df = pd.read_csv("data/vsfc_sent/train.csv")
test_df = pd.read_csv("data/vsfc_sent/test.csv")

# 2. Tính TF-IDF char_wb (3-5 gram)
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
X_train = vectorizer.fit_transform(train_df['text'].astype(str))
X_test = vectorizer.transform(test_df['text'].astype(str))

# 3. Tìm các mẫu test có cosine >= 0.95 với train
sim_matrix = cosine_similarity(X_test, X_train)
max_sims = sim_matrix.max(axis=1)
duplicate_indices = np.where(max_sims >= 0.95)[0]

print(f"Tìm thấy {len(duplicate_indices)} mẫu test bị trùng (>= 0.95).")
print(f"Tỷ lệ: {len(duplicate_indices)/len(test_df)*100:.2f}%")

# 4. Lấy nhãn và logits
npz_files = glob.glob(f"{LOGITS_DIR}/*.npz")
data = np.load(npz_files[0])
y_true = data["eval_gold"]
n_samples = len(y_true)

logits_data = {r: [] for r in RANKS}
for r in RANKS:
    files = glob.glob(f"{LOGITS_DIR}/R1__s*_global__L-1__r{r}__e6*.npz")
    for f in files:
        logits_data[r].append(np.load(f)["eval_logits"])

# 5. Lọc bỏ các mẫu trùng và tính F1
clean_indices = [i for i in range(n_samples) if i not in set(duplicate_indices)]
print(f"Số mẫu test sạch còn lại: {len(clean_indices)}")

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
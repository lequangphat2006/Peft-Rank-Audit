"""Check near-duplicate leakage train <-> cal / test, cho mọi task.

Usage:
  python code/check_near_dup.py --data_dir data/vsfc_sent
  python code/check_near_dup.py --data_dir /kaggle/input/peft-data \
      --train train.csv --cal calibration.csv --test test.csv
"""
import argparse, json, os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def check(train_csv, cal_csv, test_csv, thrs=(0.90, 0.95, 0.98)):
    tr  = pd.read_csv(train_csv)
    cal = pd.read_csv(cal_csv)
    te  = pd.read_csv(test_csv)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200000)
    X_tr = vec.fit_transform(tr["text"])
    out = {"n_train": len(tr), "n_cal": len(cal), "n_test": len(te), "thresholds": list(thrs)}
    for name, df in [("calibration", cal), ("test", te)]:
        X = vec.transform(df["text"])
        mx = cosine_similarity(X, X_tr).max(axis=1)
        out[name] = {f"{t:.2f}": float((mx >= t).mean()) for t in thrs}
        out[name]["max_sim_mean"] = float(mx.mean())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--train", default="train.csv")
    ap.add_argument("--cal",   default="calibration.csv")
    ap.add_argument("--test",  default="test.csv")
    ap.add_argument("--out",   default=None)
    a = ap.parse_args()
    r = check(os.path.join(a.data_dir, a.train),
              os.path.join(a.data_dir, a.cal),
              os.path.join(a.data_dir, a.test))
    print(json.dumps(r, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(r, open(a.out, "w"), indent=1)
        print(f"\nSaved: {a.out}")
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""Convert the pre-split UIT-VSFC sentiment files (train / val / test) to the layout run_pilot.py expects.

Output: <out_dir>/{train,calibration,test}.csv with columns text,label (label = 0 negative, 1 neutral, 2 positive).
 - val  -> calibration.csv  (used to pick LR / epochs now, and for temperature scaling in Paper B)
 - text column: text_clean for XLM-R / RoBERTa ; text_seg (word-segmented) for PhoBERT.
Also prints class counts, near-duplicate leakage across splits, and the largest N that supports rho=1 (gate G0).

  python prepare_vsfc.py --train train.csv --val val.csv --test test.csv --out_dir data/vsfc_sent
  python prepare_vsfc.py ... --text_col text_seg --out_dir data/vsfc_sent_phobert
"""
import argparse, os
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import rankcap


def main():
    ap = argparse.ArgumentParser()
    for k in ("train", "val", "test", "out_dir"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--text_col", default="text_clean", choices=["text", "text_clean", "text_seg"])
    ap.add_argument("--label_col", default="label_id")
    ap.add_argument("--thr", type=float, default=0.95)
    a = ap.parse_args()
    sp = {n: pd.read_csv(getattr(a, n), encoding="utf-8-sig") for n in ("train", "val", "test")}
    os.makedirs(a.out_dir, exist_ok=True)
    for n, d in sp.items():
        assert d[a.text_col].notna().all() and (d[a.text_col].astype(str).str.strip() != "").all(), f"empty text in {n}"
        assert set(d[a.label_col].unique()) <= {0, 1, 2}, f"unexpected labels in {n}"
        out = d[[a.text_col, a.label_col]].rename(columns={a.text_col: "text", a.label_col: "label"})
        out.to_csv(os.path.join(a.out_dir, {"val": "calibration"}.get(n, n) + ".csv"), index=False)
    if "group_id" in sp["train"]:
        for x, y in (("train", "val"), ("train", "test"), ("val", "test")):
            assert not (set(sp[x].group_id) & set(sp[y].group_id)), f"group_id leak {x}/{y}"
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit(
        pd.concat([d[a.text_col] for d in sp.values()]))
    Xtr = vec.transform(sp["train"][a.text_col])
    for n in ("val", "test"):
        X = vec.transform(sp[n][a.text_col]); hit = 0
        for s in range(0, X.shape[0], 500):
            hit += ((X[s:s + 500] @ Xtr.T).toarray().max(1) >= a.thr).sum()
        print(f"{n}: {hit}/{len(sp[n])} rows ({hit / len(sp[n]):.1%}) have a near-duplicate (cos>={a.thr}) in train")
    cnt = np.bincount(sp["train"][a.label_col], minlength=3)
    print("sizes:", {n: len(d) for n, d in sp.items()}, "| train class counts [neg,neu,pos]:", cnt.tolist())
    N = rankcap.max_feasible_N(cnt)
    print("G0: max N for rho=1 =", N)
    for n_try in (1500, 1200, 900):
        if n_try <= N:
            print(f"    N={n_try}: feasible rho -> {rankcap.feasible_imbalance(cnt, n_try)}; "
                  f"per-class need at rho=20: {rankcap.imbalanced_counts(3, n_try, 20).tolist()}")
            break


if __name__ == "__main__":
    main()

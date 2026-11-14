"""ext_confirmatory.py - họ xác nhận của EXT-1 (đã fix hoàn chỉnh)"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_common import (read_csv, load_lr_map, load_second_best, select_branch, select_fixed_lr,
                        per_seed_effects, welch, perm_p, one_sample, tost_from, onesided_pos_p, holm,
                        macro_f1_fast)

def load_many(paths):
    if not paths: return pd.DataFrame()
    if isinstance(paths, str): paths = [paths]
    df = pd.concat([read_csv(p) for p in paths], ignore_index=True)
    return df.drop_duplicates("run_id", keep="first")

def e1_from(dp, dx, ranks, y="macro_f1"):
    dp = dp.drop_duplicates(subset=['seed', 'rank'], keep='first')
    dx = dx.drop_duplicates(subset=['seed', 'rank'], keep='first')
    sp, drop_p = per_seed_effects(dp, y, ranks)
    sx, drop_x = per_seed_effects(dx, y, ranks)
    if len(sp) < 3 or len(sx) < 3:
        return None
    w = welch(sp["total"], sx["total"])
    w.update(total_p=float(sp["total"].mean()), total_x=float(sx["total"].mean()),
             sd_p=float(sp["total"].std(ddof=1)), sd_x=float(sx["total"].std(ddof=1)),
             dropped_p=drop_p, dropped_x=drop_x, seeds_p=sp["seed"].tolist(), seeds_x=sx["seed"].tolist())
    return w, sp, sx

def mixed_check(dp, dx, ranks):
    try:
        import warnings
        import statsmodels.formula.api as smf
        warnings.simplefilter("ignore")
        d = pd.concat([dp.assign(model="P"), dx.assign(model="X")])
        d = d[d["rank"].isin(ranks)].copy()
        d["lr2"] = np.log2(d["rank"])
        d["uid"] = d["model"] + d["seed"].astype(str)
        m = smf.mixedlm("macro_f1 ~ lr2 * C(model)", d, groups=d["uid"]).fit(reml=True)
        k = [i for i in m.params.index if "lr2:" in i][0]
        span = float(np.log2(max(ranks)) - np.log2(min(ranks)))
        return dict(coef=float(m.params[k]) * span * (-1 if "T.X" in k else 1), p=float(m.pvalues[k]), term=k)
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- bootstrap
def _load_preds(df, logits_dir, ranks):
    preds, gold, seeds = {}, None, None
    for r in ranks:
        g = df[df["rank"] == r].sort_values("seed")
        arr = []
        for _, row in g.iterrows():
            run_id = row["run_id"]
            filepath = os.path.join(logits_dir, run_id + ".npz")
            if not os.path.exists(filepath) and "T2" in run_id:
                alt_id = run_id.replace("R1__", "T2_XLM__").replace("B1X__", "T2_PHO__")
                filepath = os.path.join(logits_dir, alt_id + ".npz")
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Không tìm thấy logits: {filepath}")
                
            z = np.load(filepath)
            gl = z["eval_gold"].astype(int)
            if gold is None: 
                gold = gl
            elif not np.array_equal(gold, gl): 
                raise ValueError("nhãn gold khác nhau giữa các lượt")
            arr.append(z["eval_logits"].argmax(1).astype(np.int8))
        
        # --- ĐOẠN CODE ĐÃ ĐƯỢC THÊM VÀO ĐỂ GÁN SEEDS ---
        s = g["seed"].astype(int).tolist()
        if seeds is None:
            seeds = s
        elif s != seeds:
            raise ValueError(f"tập seed không đồng nhất giữa các rank: rank {r}")
        # ------------------------------------------------
            
        preds[r] = np.stack(arr)
    return preds, gold, seeds

def two_level_bootstrap(dp, dx, lp, lx, ranks, n_boot=2000, seed=12345):
    Pp, gp, sp = _load_preds(dp, lp, ranks)
    Px, gx, sx = _load_preds(dx, lx, ranks)
    if not np.array_equal(gp, gx): 
        raise ValueError("gold của hai mô hình khác nhau - không thể ghép cặp")
    N, K = len(gp), 3
    x = np.log2(np.asarray(ranks, float))
    w = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
    span = x.max() - x.min()

    def F_matrix(P, idx):
        S = next(iter(P.values())).shape[0]
        g = gp[idx] if idx is not None else gp
        F = np.empty((S, len(ranks)))
        for j, r in enumerate(ranks):
            for s in range(S):
                pr = P[r][s][idx] if idx is not None else P[r][s]
                F[s, j] = macro_f1_fast(g, pr, K)
        return F

    def total(F, draw=None):
        t = (F @ w) * span
        return float(t[draw].mean()) if draw is not None else float(t.mean())

    rng = np.random.default_rng(seed)
    Sp, Sx = len(sp), len(sx)
    Fp0, Fx0 = F_matrix(Pp, None), F_matrix(Px, None)
    point = dict(P=total(Fp0), X=total(Fx0))
    point["delta"] = point["P"] - point["X"]
    both, test_only, seed_only = [], [], []
    
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        Fp, Fx = F_matrix(Pp, idx), F_matrix(Px, idx)
        dp_, dx_ = rng.integers(0, Sp, Sp), rng.integers(0, Sx, Sx)
        both.append((total(Fp, dp_), total(Fx, dx_)))
        test_only.append((total(Fp), total(Fx)))
        seed_only.append((total(Fp0, dp_), total(Fx0, dx_)))

    def summ(arr):
        a = np.asarray(arr)
        d = a[:, 0] - a[:, 1]
        pc = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        return dict(ci95_P=pc(a[:, 0]), ci95_X=pc(a[:, 1]), ci95_delta=pc(d),
                    excludes_zero=bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0))
                    
    return dict(n_boot=n_boot, n_test=int(N), n_seeds_P=Sp, n_seeds_X=Sx, point=point,
                two_level=summ(both), test_only=summ(test_only), seed_only=summ(seed_only))

# ------------------------------------------------------------------ chính
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_p", required=True)
    ap.add_argument("--lr_map_p", required=True)
    ap.add_argument("--decision_p", default=None)
    ap.add_argument("--csv_x", nargs="+", required=True)
    ap.add_argument("--lr_map_x", required=True)
    ap.add_argument("--decision_x", default=None)
    
    ap.add_argument("--csv_fixed_p", default=None)
    ap.add_argument("--csv_fixed_x", default=None)

    ap.add_argument("--ranks", type=int, nargs="*", default=[1, 4, 8, 16, 32])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--fixed_lr", type=float, default=2e-4)
    ap.add_argument("--logits_p", default=None)
    ap.add_argument("--logits_x", default=None)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dp_all = read_csv(a.csv_p)
    dp_all = dp_all[(dp_all["epochs"] == a.epochs) & dp_all["rank"].isin(a.ranks)]
    dx_all = load_many(a.csv_x)
    dx_all = dx_all[(dx_all["epochs"] == a.epochs) & dx_all["rank"].isin(a.ranks)]
    dp = select_branch(dp_all, load_lr_map(a.lr_map_p))
    dx = select_branch(dx_all, load_lr_map(a.lr_map_x))
    out = dict(ranks=a.ranks, epochs=a.epochs, delta=a.delta)

    # ---- E1
    r1 = e1_from(dp, dx, a.ranks)
    if r1 is None: 
        raise SystemExit("Chưa đủ seed hoàn chỉnh.")
    e1, sp, sx = r1
    e1["perm_p"] = perm_p(sp["total"], sx["total"])
    e1["mixed_model"] = mixed_check(dp, dx, a.ranks)
    
    # ---- E2, E3
    m, se, df = one_sample(sp["total"])
    p_e2, ci90 = tost_from(m, se, df, a.delta)
    nn, _ = per_seed_effects(dp, "nll", a.ranks)
    mn, sen, dfn = one_sample(nn["total"])
    p_e3 = onesided_pos_p(mn, sen, dfn)
    adj = holm([e1["p"], p_e2, p_e3])
    out.update(E1=dict(**e1, p_holm=adj[0]),
               E2=dict(est=m, se=se, df=df, ci90=ci90, p=p_e2, p_holm=adj[1], equivalent=bool(adj[1] < 0.05)),
               E3=dict(est=mn, se=sen, df=dfn, p=p_e3, p_holm=adj[2], increases=bool(adj[2] < 0.05)))

    print(f"\n=== HỌ XÁC NHẬN EXT-1 (Holm m=3) | PhoBERT {len(sp)} seed, XLM-R {len(sx)} seed ===")
    print(f"E1 tương tác: Delta = {e1['diff']:+.4f} CI95 [{e1['ci95'][0]:+.4f},{e1['ci95'][1]:+.4f}] "
          f"p_holm={adj[0]:.4g} | hoán vị p={e1['perm_p']:.4g}")
    print(f"E2 PhoBERT TOST: tổng {m:+.4f} CI90 [{ci90[0]:+.4f},{ci90[1]:+.4f}] p_holm={adj[1]:.4g}")
    print(f"E3 PhoBERT NLL: {mn:+.4f} p_holm={adj[2]:.4g}")

    # ---- bootstrap
    boot = None
    if a.logits_p and a.logits_x:
        try:
            dp_boot = dp[dp['cell'].astype(str).str.contains('T2_PHO', na=False)]
            dx_boot = dx[dx['cell'].astype(str).str.contains('T2_XLM', na=False)]
            boot = two_level_bootstrap(dp_boot, dx_boot, a.logits_p, a.logits_x, a.ranks, a.n_boot)
            out["bootstrap"] = boot
            t = boot["two_level"]
            print(f"\nBootstrap: Delta {boot['point']['delta']:+.4f} CI95 [{t['ci95_delta'][0]:+.4f},{t['ci95_delta'][1]:+.4f}]")
        except Exception as e:
            print(f"\nBỎ QUA bootstrap: {type(e).__name__}: {e}")
            out["bootstrap_error"] = f"{type(e).__name__}: {e}"

    # ---- nhánh độ bền
    rob = {}
    
    # 1. Fixed LR
    dp_fix_src = read_csv(a.csv_fixed_p) if a.csv_fixed_p else dp_all
    dx_fix_src = load_many(a.csv_fixed_x) if a.csv_fixed_x else dx_all
    dfp_fixed = select_fixed_lr(dp_fix_src, a.fixed_lr)
    dfx_fixed = select_fixed_lr(dx_fix_src, a.fixed_lr)
    r = e1_from(dfp_fixed, dfx_fixed, a.ranks)
    if r: rob["fixed_lr"] = {k: r[0][k] for k in ("diff", "ci95", "p", "seeds_p", "seeds_x")}
        
    # 2. Second-best LR
    if a.decision_p and a.decision_x and os.path.exists(a.decision_p):
        try:
            dp_sec_src = dp_all[dp_all['cell'].astype(str).str.contains('SECOND', na=False)]
            dx_sec_src = dx_all[dx_all['cell'].astype(str).str.contains('SECOND', na=False)]
            if not dp_sec_src.empty and not dx_sec_src.empty:
                dp_second = select_branch(dp_sec_src, load_second_best(a.decision_p))
                dx_second = select_branch(dx_sec_src, load_second_best(a.decision_x))
                r_sec = e1_from(dp_second, dx_second, a.ranks)
                if r_sec: rob["second_best_lr"] = {k: r_sec[0][k] for k in ("diff", "ci95", "p", "seeds_p", "seeds_x")}
        except Exception as e: 
            print(f"Lỗi second_best: {e}")
            
    out["robustness_E1"] = rob
    print("\nĐộ bền của E1:")
    for k, v in rob.items():
        print(f"  {k}: Delta {v['diff']:+.4f} CI95 [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] p={v['p']:.4g}")

    # ---- quy tắc kết luận
    sign = np.sign(e1["diff"])
    cond = dict(p_holm_E1_lt_0_05=bool(adj[0] < 0.05),
                bootstrap_ci_excludes_zero=(boot["two_level"]["excludes_zero"] if boot else None),
                sign_agrees_fixed_lr=(bool(np.sign(rob["fixed_lr"]["diff"]) == sign) if "fixed_lr" in rob else None),
                sign_agrees_second_best=(bool(np.sign(rob["second_best_lr"]["diff"]) == sign) if "second_best_lr" in rob else None))
    
    known = [v for v in cond.values() if v is not None]
    pending = [k for k, v in cond.items() if v is None]
    if known and not all(known): 
        verdict = "CHƯA ĐỦ BẰNG CHỨNG cho 'phụ thuộc mô hình'"
    elif pending: 
        verdict = "CHƯA THỂ KẾT LUẬN: còn thiếu " + ", ".join(pending)
    else: 
        verdict = "ĐƯỢC PHÉP viết 'ảnh hưởng của rank phụ thuộc mô hình'"
    
    out["decision_rule"] = dict(conditions=cond, verdict=verdict)
    print(f"\nQuy tắc kết luận: {verdict}\n  điều kiện: {cond}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
        print(f"Đã lưu {a.out}")

if __name__ == "__main__":
    main()
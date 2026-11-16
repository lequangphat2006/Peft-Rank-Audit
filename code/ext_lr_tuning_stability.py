"""Gap analysis + session effect + git provenance, dung schema thuc te."""
import os, csv, json
import numpy as np
from collections import defaultdict

def load_csv(p):
    if not os.path.exists(p): return []
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

def xlmr_gap():
    rows = load_csv("results/tuning/tuning_full.csv")
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        try:
            rank = int(r["rank"]); lr = float(r["lr"]); f1 = float(r["cal_macro_f1"])
        except (ValueError, KeyError): continue
        groups[rank][lr].append(f1)
    print("="*76)
    print("XLM-R: gap best vs second-best (calibration Macro-F1, tuning_full.csv)")
    print("="*76)
    print(f"{'rank':>4} {'best_lr':>9} {'best_mu':>8} {'best_sd':>8} "
          f"{'2nd_lr':>9} {'2nd_mu':>8} {'gap':>8} {'n':>3}")
    print("-"*76)
    for rank in sorted(groups):
        stats = sorted(
            [(lr, float(np.mean(v)),
              float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))
             for lr, v in groups[rank].items()],
            key=lambda x: -x[1])
        if len(stats) < 2:
            lr1, mu1, sd1, n1 = stats[0]
            print(f"{rank:>4} {lr1:>9.4g} {mu1:>8.4f} {sd1:>8.4f} "
                  f"{'-':>9} {'-':>8} {'-':>8} {n1:>3}")
            continue
        (lr1, mu1, sd1, n1), (lr2, mu2, _, _) = stats[0], stats[1]
        gap = mu1 - mu2
        flag = " <<TIE" if gap < 0.005 else ""
        print(f"{rank:>4} {lr1:>9.4g} {mu1:>8.4f} {sd1:>8.4f} "
              f"{lr2:>9.4g} {mu2:>8.4f} {gap:>8.4f} {n1:>3}{flag}")

def pho_gap():
    with open("results/ext/phobert/lr_map_best.json", encoding="utf-8") as fh:
        d = json.load(fh)
    print("\n" + "="*76)
    print("PhoBERT: gap best vs second-best (lr_map_best.json)")
    print("="*76)
    print(f"{'rank':>4} {'best_lr':>9} {'best_mu':>8} {'2nd_lr':>9} "
          f"{'2nd_mu':>8} {'gap':>8} {'extended':>9}")
    print("-"*76)
    for rank in sorted(d.keys(), key=int):
        info = d[rank]
        t = info["table"]
        best, second = t[0], t[1]
        gap = best["mean_f1"] - second["mean_f1"]
        flag = " <<TIE" if gap < 0.005 else ""
        print(f"{rank:>4} {best['lr']:>9.4g} {best['mean_f1']:>8.4f} "
              f"{second['lr']:>9.4g} {second['mean_f1']:>8.4f} {gap:>8.4f} "
              f"{str(info.get('extended', False)):>9}{flag}")

def session_effect():
    orig = load_csv("results/ext/phobert/tune_all_v2.csv")
    ctrl = load_csv("results/ext/phobert/tune_ctrl_1e4.csv")
    orig_by_rank = defaultdict(dict)
    for r in orig:
        try:
            rank = int(r["rank"]); lr = float(r["lr"]); f1 = float(r["cal_macro_f1"])
            seed = int(r["seed"])
        except (ValueError, KeyError): continue
        if abs(lr - 1e-4) < 1e-10 and rank in (8, 16, 32):
            orig_by_rank[rank][seed] = f1
    ctrl_by_rank = defaultdict(dict)
    for r in ctrl:
        try:
            rank = int(r["rank"]); lr = float(r["lr"]); f1 = float(r["cal_macro_f1"])
            seed = int(r["seed"])
        except (ValueError, KeyError): continue
        if abs(lr - 1e-4) < 1e-10:
            ctrl_by_rank[rank][seed] = f1
    print("\n" + "="*76)
    print("SESSION EFFECT: PhoBERT at LR=1e-4 (original vs control session)")
    print("="*76)
    print(f"{'rank':>4} {'orig_mu':>8} {'ctrl_mu':>8} {'diff':>8} {'sd_diff':>8} {'n':>3}")
    print("-"*76)
    for rank in [8, 16, 32]:
        o = orig_by_rank.get(rank, {}); c = ctrl_by_rank.get(rank, {})
        seeds = sorted(set(o) & set(c))
        if not seeds:
            print(f"{rank:>4} {'-':>8} {'-':>8} {'-':>8} {'-':>8} {'-':>3}")
            continue
        ovals = [o[s] for s in seeds]; cvals = [c[s] for s in seeds]
        diffs = [o[s] - c[s] for s in seeds]
        sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        print(f"{rank:>4} {np.mean(ovals):>8.4f} {np.mean(cvals):>8.4f} "
              f"{np.mean(diffs):>+8.4f} {sd:>8.4f} {len(seeds):>3}")

def git_provenance():
    files = [
        "results/tuning/tuning_full.csv",
        "results/ext/phobert/tune_all.csv",
        "results/ext/phobert/tune_all_v2.csv",
        "results/ext/phobert/tune_ctrl_1e4.csv",
        "results/tuning/tuning_5e5.csv",
    ]
    print("\n" + "="*76)
    print("GIT PROVENANCE (for reproducibility)")
    print("="*76)
    for f in files:
        rows = load_csv(f)
        if not rows: continue
        gits = sorted(set(r.get("git", "?") for r in rows))
        print(f"  {f}: {gits}")

if __name__ == "__main__":
    xlmr_gap()
    pho_gap()
    session_effect()
    git_provenance()

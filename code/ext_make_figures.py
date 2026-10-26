"""ext_make_figures.py - 3 hình cho phần mở rộng EXT-1 (cùng phong cách make_figures.py: PNG + PDF).

  fig13_two_model_f1_vs_rank : Macro-F1 theo rank, hai mô hình, trung bình +- CI 95% theo seed (nhánh LR theo rank)
  fig14_interaction_forest   : hiệu ứng rank 1->32 của từng mô hình và hiệu (tương tác), CI seed và CI bootstrap hai tầng
  fig15_phobert_lr_landscape : F1 calibration theo (rank, LR) của vòng tune, đánh dấu LR được chọn và lượt sập

  python code/ext_make_figures.py --csv_p ... --lr_map_p ... --csv_x ... --lr_map_x ... \
      --tune_p results/ext/phobert/tune_all.csv --decision_p results/ext/phobert/tuning_decision.json \
      --confirmatory results/ext/phobert/confirmatory.json --outdir figures
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_common import read_csv, load_lr_map, select_branch  # noqa: E402

COL = {"PhoBERT": "#c0392b", "XLM-R": "#1f4e79"}


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def fig13(a):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for tag, name, off in (("p", "PhoBERT", -0.03), ("x", "XLM-R", 0.03)):
        paths = getattr(a, f"csv_{tag}")
        if not paths:
            continue
        df = pd.concat([read_csv(p) for p in paths]).drop_duplicates("run_id")
        df = select_branch(df[(df["epochs"] == 6) & df["rank"].isin(a.ranks)], load_lr_map(getattr(a, f"lr_map_{tag}")))
        g = df.groupby("rank")["macro_f1"]
        m, n, sd = g.mean(), g.count(), g.std(ddof=1)
        ci = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        x = np.log2(m.index.values.astype(float)) + off
        ax.errorbar(x, m.values, yerr=ci.values, marker="o", capsize=3, color=COL[name],
                    label=f"{name} (n={int(n.min())}-{int(n.max())} seed)")
    ax.set_xticks(np.log2(a.ranks)); ax.set_xticklabels(a.ranks)
    ax.set_xlabel("Rank LoRA (thang log2)"); ax.set_ylabel("Macro-F1 (test)")
    ax.set_title("Macro-F1 theo rank: hai mô hình (CI 95% theo seed)")
    ax.grid(alpha=.3); ax.legend()
    save(fig, a.outdir, "fig13_two_model_f1_vs_rank")


def fig14(a):
    if not a.confirmatory or not os.path.exists(a.confirmatory):
        print("bỏ qua fig14: thiếu --confirmatory"); return
    c = json.load(open(a.confirmatory, encoding="utf-8"))
    e1 = c["E1"]
    def ci(mean, sd, n):
        h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        return [mean - h, mean + h]
    rows = [("PhoBERT", e1["total_p"], ci(e1["total_p"], e1["sd_p"], e1["n_a"])),
            ("XLM-R", e1["total_x"], ci(e1["total_x"], e1["sd_x"], e1["n_b"])),
            ("Tương tác (P - X)", e1["diff"], e1["ci95"])]
    boot = c.get("bootstrap")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, (lab, est, cint) in enumerate(rows):
        y = 2 - i
        col = COL.get(lab, "#333333")
        ax.plot(cint, [y, y], color=col, lw=2.2, solid_capstyle="round",
                label="CI 95% theo seed" if i == 2 else None)
        ax.plot([est], [y], "o", color=col, ms=7)
        if lab.startswith("Tương tác") and boot:
            ax.plot(boot["two_level"]["ci95_delta"], [y - .22, y - .22], color="#7f8c8d", lw=2.2,
                    label="CI 95% bootstrap hai tầng (test + seed)")
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels([r[0] for r in rows]); ax.set_ylim(-0.6, 2.5)
    ax.set_xlabel("Tổng hiệu ứng rank 1→32 lên Macro-F1"); ax.set_title("Phụ thuộc mô hình: hiệu ứng rank và tương tác")
    ax.grid(alpha=.3, axis="x")
    h, l = ax.get_legend_handles_labels()
    if h:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8, ncol=2, frameon=False)
    save(fig, a.outdir, "fig14_interaction_forest")


def fig15(a):
    if not a.tune_p:
        print("bỏ qua fig15: thiếu --tune_p"); return
    t = read_csv(a.tune_p)
    t = t[t["config"] == "global"] if "config" in t.columns else t
    piv = t.groupby(["rank", "lr"])["macro_f1"].mean().unstack("lr").sort_index()
    crash = t.assign(c=t["macro_f1"] < 0.25).groupby(["rank", "lr"])["c"].mean().unstack("lr").reindex(index=piv.index, columns=piv.columns)
    best = {}
    if a.decision_p and os.path.exists(a.decision_p):
        best = load_lr_map(a.decision_p)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0.6, vmax=0.84)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f"{x:g}" for x in piv.columns])
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    for i, r in enumerate(piv.index):
        for j, lr in enumerate(piv.columns):
            v = piv.values[i, j]
            if np.isnan(v):
                continue
            s = f"{v:.3f}" + (f"\nsập {crash.values[i, j]:.0%}" if crash.values[i, j] > 0 else "")
            ax.text(j, i, s, ha="center", va="center", fontsize=7, color="w" if v < 0.76 else "k",
                    fontweight="bold" if np.isclose(best.get(int(r), -1), lr) else None)
    ax.set_xlabel("Learning rate"); ax.set_ylabel("Rank"); ax.set_title("PhoBERT: F1 calibration theo (rank, LR); đậm = LR được chọn")
    fig.colorbar(im, ax=ax, label="Macro-F1 calibration")
    save(fig, a.outdir, "fig15_phobert_lr_landscape")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv_p", nargs="*"); ap.add_argument("--lr_map_p")
    ap.add_argument("--csv_x", nargs="*"); ap.add_argument("--lr_map_x")
    ap.add_argument("--tune_p"); ap.add_argument("--decision_p"); ap.add_argument("--confirmatory")
    ap.add_argument("--ranks", type=int, nargs="*", default=[1, 4, 8, 16, 32])
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--update_manifest", action="store_true")
    a = ap.parse_args()
    if a.csv_p or a.csv_x:
        fig13(a)
    fig14(a); fig15(a)
    if a.update_manifest:
        mp = os.path.join(a.outdir, "manifest.json")
        m = json.load(open(mp, encoding="utf-8-sig")) if os.path.exists(mp) else {}
        m["fig13_two_model_f1_vs_rank"] = ["B1_ext_merged.csv", "global_merged.csv"]
        m["fig14_interaction_forest"] = ["confirmatory.json"]
        m["fig15_phobert_lr_landscape"] = ["tune_all.csv", "tuning_decision.json"]
        json.dump(m, open(mp, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()

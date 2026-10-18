#!/usr/bin/env python3
"""make_figures.py - Tạo toàn bộ hình cho bài báo PEFT Rank Audit.

Chạy từ thư mục gốc của repo (hoặc dùng --root):

    python code/make_figures.py                       # tạo tất cả
    python code/make_figures.py --only fig03 fig05    # chỉ một số hình
    python code/make_figures.py --no-logits           # bỏ các hình cần logits
    python code/make_figures.py --formats png pdf     # định dạng xuất

Mỗi hình đọc đúng các file kết quả có trong repo, không có số liệu gõ tay.
Đầu ra: figures/figNN_*.{png,pdf} và figures/manifest.json (nguồn dữ liệu từng hình).

Phụ thuộc: numpy, pandas, matplotlib, scipy (đã có trong requirements.txt).
Fig 03 (dòng near-duplicate) cần thêm scikit-learn và data/vsfc_sent/{train,test}.csv;
nếu không có thì dòng đó tự động bị bỏ qua.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy import stats
from scipy.optimize import minimize_scalar

# ----------------------------------------------------------------------------
# Cấu hình chung
# ----------------------------------------------------------------------------
RANKS = [1, 4, 8, 16, 32]
DELTA_F1 = 0.01
DELTA_NLL = 0.01
COLLAPSE_F1 = 0.25
N_BOOT = 1000
RNG_SEED = 42

C = {  # bảng màu Okabe-Ito, an toàn cho người mù màu
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
    "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#7F7F7F", "black": "#000000",
}
EP_COLOR = {3: C["orange"], 6: C["blue"]}

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

MANIFEST = {}


# ----------------------------------------------------------------------------
# Tiện ích dữ liệu
# ----------------------------------------------------------------------------
class Ctx:
    def __init__(self, root, outdir, formats, use_logits):
        self.root, self.outdir, self.formats, self.use_logits = root, outdir, formats, use_logits
        self._cache = {}

    def p(self, *a):
        return os.path.join(self.root, *a)

    def csv(self, *a):
        key = os.path.join(*a)
        if key not in self._cache:
            path = self.p(*a)
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            self._cache[key] = pd.read_csv(path)
        return self._cache[key].copy()

    def json(self, *a):
        with open(self.p(*a), encoding="utf-8") as f:
            return json.load(f)

    def save(self, fig, name, sources):
        os.makedirs(self.outdir, exist_ok=True)
        for ext in self.formats:
            fig.savefig(os.path.join(self.outdir, f"{name}.{ext}"))
        plt.close(fig)
        MANIFEST[name] = sources
        print(f"  [ok] {name}  <- {', '.join(sources)}")


def mean_ci(x, level=0.95):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    if n < 2:
        return m, 0.0
    h = stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, h


def total_effect(df, y, level=0.90):
    """Tổng hiệu ứng rank 1->32 (5 x độ dốc theo log2 rank), có hiệu ứng cố định theo seed.
    Trả về (ước lượng, cận dưới, cận trên) theo CI `level`."""
    d = df.copy()
    X = pd.get_dummies(d["seed"], drop_first=True).astype(float)
    X.insert(0, "c", 1.0)
    X["l2"] = np.log2(d["rank"].values)
    Xv, yv = X.values, d[y].values
    beta = np.linalg.lstsq(Xv, yv, rcond=None)[0]
    res = yv - Xv @ beta
    dof = len(yv) - Xv.shape[1]
    s2 = res @ res / dof
    cov = s2 * np.linalg.inv(Xv.T @ Xv)
    i = list(X.columns).index("l2")
    span = np.log2(d["rank"].max()) - np.log2(d["rank"].min())
    est, se = span * beta[i], span * np.sqrt(cov[i, i])
    t = stats.t.ppf(0.5 + level / 2, dof)
    return est, est - t * se, est + t * se


def style_rank_axis(ax, lrs=None):
    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS)
    if lrs is None:
        ax.set_xticklabels([str(r) for r in RANKS])
    else:
        ax.set_xticklabels([f"{r}\n{lrs[r]:g}" if r in lrs else str(r) for r in RANKS])
    ax.minorticks_off()
    ax.set_xlabel("LoRA rank" + ("  (LR below)" if lrs else ""))


def panel_label(ax, s):
    ax.text(-0.14, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight="bold")


# ----- logits ---------------------------------------------------------------
def load_logits(ctx, epochs=6):
    """Trả về (gold, {rank: [(seed, eval_logits, cal_logits, cal_gold), ...]})."""
    pat = ctx.p("results", "global", "logits", f"R1__s*__global__L-1__r*__e{epochs}__*.npz")
    files = sorted(glob.glob(pat))
    if not files:
        return None, None
    out, gold = {r: [] for r in RANKS}, None
    for f in files:
        m = re.search(r"__s(\d+)__global__L-1__r(\d+)__e(\d+)__", os.path.basename(f))
        if not m or int(m.group(3)) != epochs:
            continue
        seed, rank = int(m.group(1)), int(m.group(2))
        z = np.load(f)
        gold = z["eval_gold"]
        out[rank].append((seed, z["eval_logits"], z["cal_logits"], z["cal_gold"]))
    return gold, out


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll_of(logits, y, T=1.0):
    p = softmax(logits, T)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def ece_of(logits, y, T=1.0, bins=15):
    p = softmax(logits, T)
    conf, pred = p.max(1), p.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def fit_temperature(cal_logits, cal_gold):
    r = minimize_scalar(lambda T: nll_of(cal_logits, cal_gold, T), bounds=(0.05, 20), method="bounded")
    return float(r.x)


def macro_f1_batch(P, y, idx):
    """P: (S, N) nhãn dự đoán của S seed. Trả về Macro-F1 mỗi seed trên chỉ số idx."""
    yy, pp = y[idx], P[:, idx]
    f = []
    for c in range(3):
        tp = ((pp == c) & (yy == c)).sum(1)
        fp = ((pp == c) & (yy != c)).sum(1)
        fn = ((pp != c) & (yy == c)).sum(1)
        f.append(np.where(tp == 0, 0.0, 2 * tp / np.maximum(2 * tp + fp + fn, 1)))
    return np.mean(f, axis=0)


def class_f1_batch(P, y, idx):
    yy, pp = y[idx], P[:, idx]
    f = []
    for c in range(3):
        tp = ((pp == c) & (yy == c)).sum(1)
        fp = ((pp == c) & (yy != c)).sum(1)
        fn = ((pp != c) & (yy == c)).sum(1)
        f.append(np.where(tp == 0, 0.0, 2 * tp / np.maximum(2 * tp + fp + fn, 1)))
    return np.array(f)  # (3, S)


def preds_by_rank(lg):
    return {r: np.stack([np.argmax(e, 1) for (_, e, _, _) in lg[r]]) for r in RANKS if lg[r]}


def boot_total_effect(P, y, pool, n_boot=N_BOOT, seed=RNG_SEED):
    """Bootstrap ghép cặp trên mẫu test: tổng hiệu ứng rank 1->32 trên Macro-F1."""
    rng = np.random.default_rng(seed)
    ranks = sorted(P)
    x = np.log2(ranks)
    w = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
    span = x.max() - x.min()
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(pool, len(pool), replace=True)
        m = np.array([macro_f1_batch(P[r], y, idx).mean() for r in ranks])
        vals.append(span * (w @ m))
    pm = np.array([macro_f1_batch(P[r], y, pool).mean() for r in ranks])
    return span * (w @ pm), np.percentile(vals, [5, 95])


def near_dup_indices(ctx, thr=0.95):
    tr, te = ctx.p("data", "vsfc_sent", "train.csv"), ctx.p("data", "vsfc_sent", "test.csv")
    if not (os.path.exists(tr) and os.path.exists(te)):
        return None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return None
    tr_df, te_df = pd.read_csv(tr), pd.read_csv(te)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    Xt = vec.fit_transform(tr_df["text"].astype(str))
    Xe = vec.transform(te_df["text"].astype(str))
    ms = np.concatenate([cosine_similarity(Xe[i:i + 200], Xt).max(1) for i in range(0, Xe.shape[0], 200)])
    return set(np.where(ms >= thr)[0].tolist())


# ----------------------------------------------------------------------------
# FIG 01 - Sơ đồ thiết kế nghiên cứu
# ----------------------------------------------------------------------------
def fig01_design(ctx):
    n_tune = len(ctx.csv("results", "tuning", "tuning_full.csv"))
    n_r1 = len(ctx.csv("results", "global", "global_merged.csv"))
    n_b1 = len(ctx.csv("results", "v4_phobert", "B1_final_merged.csv"))
    n_b2 = len(ctx.csv("results", "global", "B2_merged.csv"))
    n_v1 = len(ctx.csv("results", "v1_fixed_lr", "V1_merged.csv")) + len(ctx.csv("results", "posthoc", "V1b_merged.csv"))
    n_v5 = len(ctx.csv("results", "v5_fp32", "V5_merged.csv")) + len(ctx.csv("results", "posthoc", "V5b_merged.csv"))
    n_al = len(ctx.csv("results", "p2_alpha", "alpha_gpu0.csv")) + len(ctx.csv("results", "p2_alpha", "alpha_gpu1.csv"))

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=8, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.25)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=C["black"], lw=1.1))

    pre_fc, pre_ec = "#DCEBF7", C["blue"]
    ph_fc, ph_ec = "#FCEBD2", C["orange"]

    box(0.1, 4.6, 2.9, 1.2, f"Stage 1 · LR tuning\n5 ranks × 6 LRs × 3 seeds\n6 epochs, calibration F1\ntie rule 0.005  ({n_tune} runs)", pre_fc, pre_ec)
    box(3.5, 4.6, 3.0, 1.2, f"Stage 2 · R1 confirmatory\nXLM-R-base, UIT-VSFC\n5 ranks × {{3,6}} ep × 16 seeds\n({n_r1} runs)", pre_fc, pre_ec, bold=False)
    box(7.0, 4.6, 2.9, 1.2, "Pre-registered tests\nH1 TOST (δ=0.01) on F1\nH2 slope on NLL\nH3 rank × epoch", pre_fc, pre_ec)
    arrow(3.0, 5.2, 3.5, 5.2)
    arrow(6.5, 5.2, 7.0, 5.2)

    box(0.1, 2.6, 2.9, 1.3, f"B1 · PhoBERT, LR tuned\n4 ranks × 8 seeds\n({n_b1} runs)\nB2 · second task, LR 2e-4\n({n_b2} runs)", ph_fc, ph_ec, fs=7.5)
    box(3.5, 2.6, 3.0, 1.3, f"V1/V1b · fixed LR 2e-4\n(H3 branch check)  ({n_v1} runs)\nV5/V5b · fp32 vs fp16  ({n_v5} runs)", ph_fc, ph_ec, fs=7.5)
    box(7.0, 2.6, 2.9, 1.3, f"α sensitivity (fixed:16)\n({n_al} runs, exploratory)\nAdaLoRA baseline\n(3 seeds, untuned)", ph_fc, ph_ec, fs=7.5)
    arrow(5.0, 4.6, 5.0, 3.9)

    box(0.1, 0.4, 9.8, 1.4,
        "Evidence reported\nF1 and NLL vs rank  ·  equivalence forest (seed CI vs test-bootstrap CI)  ·  LR landscape and collapse\n"
        "H3 sign reversal  ·  calibration (ECE, temperature scaling)  ·  per-class F1  ·  model dependence",
        "#F2F2F2", C["grey"], fs=8)
    arrow(5.0, 2.6, 5.0, 1.8)

    ax.add_patch(Rectangle((0.1, 0.02), 0.25, 0.18, fc=pre_fc, ec=pre_ec))
    ax.text(0.45, 0.11, "pre-registered (OSF)", fontsize=7.5, va="center")
    ax.add_patch(Rectangle((2.9, 0.02), 0.25, 0.18, fc=ph_fc, ec=ph_ec))
    ax.text(3.25, 0.11, "post hoc / exploratory", fontsize=7.5, va="center")
    ctx.save(fig, "fig01_design", ["tuning_full.csv", "global_merged.csv", "B1_final_merged.csv", "B2_merged.csv",
                                    "V1_merged.csv", "V1b_merged.csv", "V5_merged.csv", "V5b_merged.csv", "alpha_gpu*.csv"])


# ----------------------------------------------------------------------------
# FIG 02 - Kết quả chính R1: F1 và NLL theo rank
# ----------------------------------------------------------------------------
def fig02_main(ctx):
    d = ctx.csv("results", "global", "global_merged.csv")
    lrs = d[d.epochs == 6].groupby("rank").lr.first().to_dict()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, y, ttl in zip(axes, ["macro_f1", "nll"], ["Test Macro-F1", "Test NLL"]):
        for ep in (3, 6):
            g = d[d.epochs == ep]
            ms, hs = [], []
            for r in RANKS:
                v = g[g["rank"] == r][y].values
                m, h = mean_ci(v)
                ms.append(m)
                hs.append(h)
                ax.scatter(np.full(len(v), r) * np.exp(np.random.default_rng(r).normal(0, 0.04, len(v))), v,
                           s=5, alpha=0.25, color=EP_COLOR[ep], linewidths=0)
            ax.errorbar(RANKS, ms, yerr=hs, color=EP_COLOR[ep], marker="o", ms=4, lw=1.4, capsize=2.5,
                        label=f"{ep} epochs")
        style_rank_axis(ax, lrs)
        ax.set_ylabel(ttl)
        ax.set_title(ttl + " vs rank (R1, 16 seeds)")
    # băng δ quanh trung bình rank 1 ở 6 epoch cho F1
    m1 = d[(d.epochs == 6) & (d["rank"] == 1)].macro_f1.mean()
    axes[0].axhspan(m1 - DELTA_F1, m1 + DELTA_F1, color=C["green"], alpha=0.10, lw=0)
    axes[0].text(1.05, m1 + DELTA_F1 - 0.0015, "±δ (0.01) around rank-1, 6 ep", fontsize=7, color=C["green"], va="top")
    axes[0].legend(loc="lower right", frameon=False)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.tight_layout()
    ctx.save(fig, "fig02_main_f1_nll_vs_rank", ["global_merged.csv"])


# ----------------------------------------------------------------------------
# FIG 03 - Forest plot kiểm định tương đương
# ----------------------------------------------------------------------------
def fig03_forest(ctx):
    r1 = ctx.csv("results", "global", "global_merged.csv")
    r1 = r1[r1.epochs == 6]
    b1 = ctx.csv("results", "v4_phobert", "B1_final_merged.csv")
    b2 = ctx.csv("results", "global", "B2_merged.csv")

    rows_f1, rows_nll = [], []
    e, lo, hi = total_effect(r1, "macro_f1")
    rows_f1.append(("R1 XLM-R · seed CI (pre-registered)", e, lo, hi, C["blue"]))
    e, lo, hi = total_effect(r1, "nll")
    rows_nll.append(("R1 XLM-R · seed CI", e, lo, hi, C["blue"]))

    src = ["global_merged.csv", "B1_final_merged.csv", "B2_merged.csv"]
    if ctx.use_logits:
        gold, lg = load_logits(ctx, 6)
        if gold is not None:
            P = preds_by_rank(lg)
            est, (lo, hi) = boot_total_effect(P, gold, np.arange(len(gold)))
            rows_f1.append(("R1 · test-sample bootstrap", est, lo, hi, C["red"]))
            dup = near_dup_indices(ctx)
            if dup is not None:
                clean = np.array([i for i in range(len(gold)) if i not in dup])
                est, (lo, hi) = boot_total_effect(P, gold, clean)
                rows_f1.append((f"R1 · bootstrap, near-dup removed (n={len(clean)})", est, lo, hi, C["red"]))
            # NLL bootstrap không cần: NLL dùng CI theo seed
            src.append("results/global/logits/*.npz")
    e, lo, hi = total_effect(b1, "macro_f1")
    rows_f1.append(("B1 PhoBERT · LR tuned", e, lo, hi, C["purple"]))
    e, lo, hi = total_effect(b1, "nll")
    rows_nll.append(("B1 PhoBERT · LR tuned", e, lo, hi, C["purple"]))
    e, lo, hi = total_effect(b2, "macro_f1")
    rows_f1.append(("B2 second task · LR fixed 2e-4", e, lo, hi, C["orange"]))
    e, lo, hi = total_effect(b2, "nll")
    rows_nll.append(("B2 second task · LR fixed 2e-4", e, lo, hi, C["orange"]))

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
    for ax, rows, delta, ttl in [(axes[0], rows_f1, DELTA_F1, "Macro-F1: total effect of rank 1→32"),
                                 (axes[1], rows_nll, DELTA_NLL, "NLL: total effect of rank 1→32")]:
        ax.axvspan(-delta, delta, color=C["green"], alpha=0.12, lw=0)
        ax.axvline(0, color=C["black"], lw=0.8)
        for i, (lab, e, lo, hi, col) in enumerate(rows):
            y = len(rows) - 1 - i
            ax.plot([lo, hi], [y, y], color=col, lw=2.2)
            ax.plot(e, y, "o", color=col, ms=5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.5)
        ax.set_xlabel("effect (90% CI); shaded band = ±δ = 0.01")
        ax.set_title(ttl, fontsize=9)
        ax.grid(axis="y", visible=False)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.tight_layout()
    ctx.save(fig, "fig03_equivalence_forest", src)


# ----------------------------------------------------------------------------
# FIG 04 - Địa hình LR và hiện tượng sập
# ----------------------------------------------------------------------------
def fig04_lr_landscape(ctx):
    t = ctx.csv("results", "tuning", "tuning_full.csv")
    chosen = ctx.csv("results", "global", "global_merged.csv").groupby("rank").lr.first().to_dict()
    lrs = sorted(t.lr.unique())
    M = np.full((len(RANKS), len(lrs)), np.nan)
    K = np.zeros_like(M)
    N = np.zeros_like(M)
    for i, r in enumerate(RANKS):
        for j, lr in enumerate(lrs):
            g = t[(t["rank"] == r) & np.isclose(t.lr, lr)]
            if len(g):
                M[i, j] = g.cal_macro_f1.mean()
                K[i, j] = (g.cal_macro_f1 < COLLAPSE_F1).sum()
                N[i, j] = len(g)
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("#D9D9D9")
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0.78, vmax=0.82)
    for i in range(len(RANKS)):
        for j in range(len(lrs)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color=C["grey"])
                continue
            col = "white" if (M[i, j] < 0.79 and M[i, j] >= 0.78) else "black"
            if M[i, j] >= 0.80:
                col = "black"
            ax.text(j, i - 0.12, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7.5, color=col)
            if K[i, j] > 0:
                ax.text(j, i + 0.25, f"collapse {int(K[i, j])}/{int(N[i, j])}", ha="center", va="center",
                        fontsize=6.5, color=C["red"], fontweight="bold")
        if chosen.get(RANKS[i]) is not None:
            jj = int(np.argmin([abs(np.log(l) - np.log(chosen[RANKS[i]])) for l in lrs]))
            ax.add_patch(Rectangle((jj - 0.5, i - 0.5), 1, 1, fill=False, ec=C["orange"], lw=2.5))
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f"{l:g}" for l in lrs])
    ax.set_yticks(range(len(RANKS)))
    ax.set_yticklabels([str(r) for r in RANKS])
    ax.set_xlabel("learning rate")
    ax.set_ylabel("LoRA rank")
    ax.set_title("LR tuning surface (calibration Macro-F1, 6 epochs, 3 seeds, fp16)\norange box = LR chosen by the pre-registered rule; grey = < 0.78")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, extend="min", pad=0.02)
    cb.set_label("mean calibration Macro-F1")
    fig.tight_layout()
    ctx.save(fig, "fig04_lr_landscape", ["tuning_full.csv", "global_merged.csv"])


# ----------------------------------------------------------------------------
# FIG 05 - H3 đảo dấu giữa hai nhánh LR
# ----------------------------------------------------------------------------
def _fixed_lr_frame(ctx):
    v1 = ctx.csv("results", "v1_fixed_lr", "V1_merged.csv")
    v1b = ctx.csv("results", "posthoc", "V1b_merged.csv")
    r1 = ctx.csv("results", "global", "global_merged.csv")
    seeds = sorted(set(v1.seed) | set(v1b.seed))
    r16 = r1[(r1["rank"] == 16) & np.isclose(r1.lr, 2e-4) & r1.seed.isin(seeds)]
    d = pd.concat([v1, v1b, r16], ignore_index=True)
    return d[np.isclose(d.lr, 2e-4)]


def fig05_h3_reversal(ctx):
    main = ctx.csv("results", "global", "global_merged.csv")
    fixed = _fixed_lr_frame(ctx)
    lr_by_rank = main[main.epochs == 6].groupby("rank").lr.first()
    try:
        it = ctx.json("results", "analysis", "interaction_terms.json")
        t_main, t_fix = it["R1_LR_per_rank"], it["Fixed_LR_2e4"]
    except Exception:
        t_main = t_fix = None

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    ylo, yhi = 0.77, 0.865
    for ax, d, ttl, tt in [(axes[0], main, "LR tuned per rank\n(pre-registered, 16 seeds)", t_main),
                           (axes[1], fixed, "LR fixed at 2e-4\n(post hoc, 4 seeds)", t_fix)]:
        for ep in (3, 6):
            g = d[d.epochs == ep]
            rs = [r for r in RANKS if (g["rank"] == r).any()]
            ms = [g[g["rank"] == r].macro_f1.mean() for r in rs]
            hs = [mean_ci(g[g["rank"] == r].macro_f1.values)[1] for r in rs]
            ax.errorbar(rs, ms, yerr=hs, marker="o", ms=4, lw=1.4, capsize=2.5, color=EP_COLOR[ep],
                        label=f"{ep} epochs")
        style_rank_axis(ax)
        ax.set_ylim(ylo, yhi)
        ax.set_title(ttl, fontsize=8.5)
        if tt:
            ax.text(0.04, 0.05, f"rank effect 3 ep: {tt['effect_3ep']:+.3f}\nrank effect 6 ep: {tt['effect_6ep']:+.3f}\n"
                                f"interaction: {tt['interaction']:+.3f}", transform=ax.transAxes, fontsize=7,
                    va="bottom", bbox=dict(fc="white", ec=C["grey"], lw=0.5, alpha=0.9))
    axes[0].set_ylabel("Test Macro-F1")
    axes[0].legend(loc="upper right", frameon=False)
    axes[1].set_yticklabels([])
    ax = axes[2]
    ax.bar(range(len(RANKS)), [lr_by_rank[r] for r in RANKS], color=C["grey"])
    ax.axhline(2e-4, color=C["red"], lw=1, ls="--")
    ax.text(len(RANKS) - 0.5, 2.3e-4, "fixed 2e-4", color=C["red"], fontsize=7, ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_xticks(range(len(RANKS)))
    ax.set_xticklabels([str(r) for r in RANKS])
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("LR chosen by tuning rule")
    ax.set_title("LR confounded\nwith rank", fontsize=8.5)
    for a, s in zip(axes, "abc"):
        panel_label(a, s)
    fig.tight_layout()
    ctx.save(fig, "fig05_h3_sign_reversal", ["global_merged.csv", "V1_merged.csv", "V1b_merged.csv", "interaction_terms.json"])


# ----------------------------------------------------------------------------
# FIG 06 - Hiệu chuẩn: reliability, NLL và ECE trước/sau temperature scaling
# ----------------------------------------------------------------------------
def fig06_calibration(ctx):
    gold, lg = load_logits(ctx, 6)
    if gold is None:
        raise RuntimeError("không tìm thấy logits")
    csv = ctx.csv("results", "global", "global_merged.csv")
    csv = csv[csv.epochs == 6]
    stat = {r: {"nll": [], "nll_ts": [], "ece": [], "ece_ts": [], "T": []} for r in RANKS}
    mism = []
    for r in RANKS:
        for seed, e, c, cg in lg[r]:
            T = fit_temperature(c, cg)
            stat[r]["T"].append(T)
            stat[r]["nll"].append(nll_of(e, gold))
            stat[r]["nll_ts"].append(nll_of(e, gold, T))
            stat[r]["ece"].append(ece_of(e, gold))
            stat[r]["ece_ts"].append(ece_of(e, gold, T))
            row = csv[(csv["rank"] == r) & (csv.seed == seed)]
            if len(row):
                mism.append(abs(row.nll.iloc[0] - stat[r]["nll"][-1]))
    if mism and max(mism) > 1e-3:
        print(f"  [warn] NLL tính lại từ logits lệch CSV tối đa {max(mism):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.9))
    # (a) reliability
    ax = axes[0]
    edges = np.linspace(0, 1, 11)
    ax.plot([0, 1], [0, 1], color=C["black"], lw=0.8, ls="--")
    for r, col in [(1, C["blue"]), (32, C["red"])]:
        conf_all = np.concatenate([softmax(e).max(1) for (_, e, _, _) in lg[r]])
        acc_all = np.concatenate([(e.argmax(1) == gold).astype(float) for (_, e, _, _) in lg[r]])
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf_all > lo) & (conf_all <= hi)
            if m.sum() >= 30:
                xs.append(conf_all[m].mean())
                ys.append(acc_all[m].mean())
        ax.plot(xs, ys, "o-", color=col, ms=3.5, lw=1.2,
                label=f"rank {r} (ECE {np.mean(stat[r]['ece']):.3f})")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Reliability (6 ep, pooled seeds)", fontsize=8.5)
    ax.legend(frameon=False, loc="upper left")
    # (b) NLL
    for ax, k, ttl, yl in [(axes[1], "nll", "NLL before / after TS", "Test NLL"),
                           (axes[2], "ece", "ECE before / after TS", "Test ECE (15 bins)")]:
        for key, col, lab, ls in [(k, C["red"], "raw", "-"), (k + "_ts", C["blue"], "temperature-scaled", "-")]:
            ms = [mean_ci(stat[r][key])[0] for r in RANKS]
            hs = [mean_ci(stat[r][key])[1] for r in RANKS]
            ax.errorbar(RANKS, ms, yerr=hs, marker="o", ms=4, lw=1.4, capsize=2.5, color=col, label=lab, ls=ls)
        style_rank_axis(ax)
        ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=8.5)
    axes[1].legend(frameon=False)
    for a, s in zip(axes, "abc"):
        panel_label(a, s)
    fig.tight_layout()
    ctx.save(fig, "fig06_calibration", ["results/global/logits/*.npz (eval và cal)", "global_merged.csv"])


# ----------------------------------------------------------------------------
# FIG 07 - Độ nhạy α (ghép cặp theo seed)
# ----------------------------------------------------------------------------
def fig07_alpha(ctx):
    a = pd.concat([ctx.csv("results", "p2_alpha", "alpha_gpu0.csv"), ctx.csv("results", "p2_alpha", "alpha_gpu1.csv")])
    a = a[a.alpha_mode.astype(str).str.startswith("fixed")]
    seeds = sorted(a.seed.unique())
    v = ctx.csv("results", "posthoc", "V1b_merged.csv")
    v = v[v.seed.isin(seeds) & v["rank"].isin([1, 32]) & (v.epochs == 6)]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for ax, y, ttl in [(axes[0], "nll", "NLL(rank 32) − NLL(rank 1)"), (axes[1], "macro_f1", "F1(rank 32) − F1(rank 1)")]:
        xs = {"α = 2r": v, "α = 16 (fixed)": a}
        means = []
        for k, (lab, df) in enumerate(xs.items()):
            diffs = []
            for s in seeds:
                g = df[df.seed == s].set_index("rank")[y]
                if 1 in g.index and 32 in g.index:
                    diffs.append((s, g[32] - g[1]))
            for s, d in diffs:
                ax.plot(k, d, "o", color=C["grey"], ms=4, alpha=0.7)
            means.append([d for _, d in diffs])
        for s_i, s in enumerate(seeds):
            try:
                ax.plot([0, 1], [means[0][s_i], means[1][s_i]], color=C["grey"], lw=0.8, alpha=0.6)
            except IndexError:
                pass
        mm = [np.mean(m) for m in means]
        ax.plot([0, 1], mm, "o-", color=C["red"], lw=2, ms=6, label="mean")
        ax.axhline(0, color=C["black"], lw=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(list(xs.keys()))
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel("difference")
        ax.set_title(ttl, fontsize=8.5)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle(f"α sensitivity (exploratory, n = {len(seeds)} seeds; LR 2e-4, 6 epochs; lines = same seed)", fontsize=8.5, y=1.02)
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.tight_layout()
    ctx.save(fig, "fig07_alpha_sensitivity", ["alpha_gpu0.csv", "alpha_gpu1.csv", "V1b_merged.csv"])


# ----------------------------------------------------------------------------
# FIG 08 - Phụ thuộc mô hình: Δ so với rank 1 cho ba thiết lập
# ----------------------------------------------------------------------------
def fig08_model_dependence(ctx):
    r1 = ctx.csv("results", "global", "global_merged.csv")
    r1 = r1[r1.epochs == 6]
    sets = [("XLM-R · UIT-VSFC (R1, LR tuned)", r1, C["blue"]),
            ("PhoBERT · UIT-VSFC (B1, LR tuned)", ctx.csv("results", "v4_phobert", "B1_final_merged.csv"), C["purple"]),
            ("B2 · second task (LR fixed 2e-4)", ctx.csv("results", "global", "B2_merged.csv"), C["orange"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, y, ttl in [(axes[0], "macro_f1", "ΔMacro-F1 vs rank 1"), (axes[1], "nll", "ΔNLL vs rank 1")]:
        for off, (lab, d, col) in enumerate(sets):
            rs = sorted(d["rank"].unique())
            base = d[d["rank"] == 1].set_index("seed")[y]
            ms, hs = [], []
            for r in rs:
                g = d[d["rank"] == r].set_index("seed")[y]
                diff = (g - base).dropna().values
                m, h = mean_ci(diff)
                ms.append(m)
                hs.append(h)
            xs = np.array(rs) * (1 + 0.05 * (off - 1))
            ax.errorbar(xs, ms, yerr=hs, marker="o", ms=4, lw=1.3, capsize=2.5, color=col, label=lab)
        ax.axhline(0, color=C["black"], lw=0.8)
        ax.axhspan(-0.01, 0.01, color=C["green"], alpha=0.10, lw=0)
        style_rank_axis(ax)
        ax.set_ylabel(ttl)
        ax.set_title(ttl + " (paired by seed, 95% CI)", fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    fig.tight_layout()
    ctx.save(fig, "fig08_model_dependence", ["global_merged.csv", "B1_final_merged.csv", "B2_merged.csv"])


# ----------------------------------------------------------------------------
# FIG 09 - F1 từng lớp với CI bootstrap trên mẫu test
# ----------------------------------------------------------------------------
def fig09_per_class(ctx):
    gold, lg = load_logits(ctx, 6)
    if gold is None:
        raise RuntimeError("không tìm thấy logits")
    P = preds_by_rank(lg)
    n = len(gold)
    support = np.bincount(gold, minlength=3)
    rng = np.random.default_rng(RNG_SEED)
    names = ["negative", "neutral", "positive"]
    est = {r: class_f1_batch(P[r], gold, np.arange(n)).mean(1) for r in RANKS}
    boots = {r: [] for r in RANKS}
    for _ in range(N_BOOT):
        idx = rng.choice(n, n, replace=True)
        for r in RANKS:
            boots[r].append(class_f1_batch(P[r], gold, idx).mean(1))
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    width = 0.15
    cols = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, len(RANKS)))
    for k, r in enumerate(RANKS):
        b = np.array(boots[r])
        lo, hi = np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)
        x = np.arange(3) + (k - 2) * width
        ax.errorbar(x, est[r], yerr=[est[r] - lo, hi - est[r]], fmt="o", ms=4, capsize=2, color=cols[k],
                    label=f"rank {r}")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{nm}\n(n = {s})" for nm, s in zip(names, support)])
    ax.set_ylabel("Test F1 (mean of 16 seeds)")
    ax.set_title("Per-class F1 with 95% test-sample bootstrap CI (6 epochs)")
    ax.legend(frameon=False, ncol=5, fontsize=7, loc="lower left")
    fig.tight_layout()
    ctx.save(fig, "fig09_per_class_f1", ["results/global/logits/*.npz"])


# ----------------------------------------------------------------------------
# FIG 10 - Ổn định số học: fp16 so với fp32 (ghép cặp theo seed)
# ----------------------------------------------------------------------------
def fig10_fp32(ctx):
    t = ctx.csv("results", "tuning", "tuning_full.csv")
    v5b = ctx.csv("results", "posthoc", "V5b_merged.csv")
    cfgs = sorted({(int(r), float(l)) for r, l in zip(v5b["rank"], v5b.lr)}, key=lambda x: (x[1], x[0]), reverse=True)
    seeds = sorted(v5b.seed.unique())
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    xt, xl = [], []
    for k, (r, lr) in enumerate(cfgs):
        for si, s in enumerate(seeds):
            a = t[(t["rank"] == r) & np.isclose(t.lr, lr) & (t.seed == s)]
            b = v5b[(v5b["rank"] == r) & np.isclose(v5b.lr, lr) & (v5b.seed == s)]
            if len(a) and len(b):
                ya, yb = a.cal_macro_f1.iloc[0], b.cal_macro_f1.iloc[0]
                x = k + (si - 1) * 0.12
                ax.plot([x, x], [ya, yb], color=C["grey"], lw=0.8, alpha=0.7)
                ax.plot(x, ya, "o", color=C["orange"], ms=5, label="fp16" if (k == 0 and si == 0) else None)
                ax.plot(x, yb, "s", color=C["blue"], ms=5, label="fp32" if (k == 0 and si == 0) else None)
        xt.append(k)
        xl.append(f"rank {r}\nLR {lr:g}")
    ax.axhline(COLLAPSE_F1, color=C["red"], ls="--", lw=1)
    ax.text(len(cfgs) - 0.55, COLLAPSE_F1 + 0.01, "collapse threshold (0.25)", color=C["red"], fontsize=7, ha="right")
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=7.5)
    ax.set_ylabel("calibration Macro-F1")
    ax.set_title(f"fp16 vs fp32 at the same seeds ({', '.join(map(str, seeds))}); lines connect the same seed")
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout()
    ctx.save(fig, "fig10_fp16_vs_fp32", ["tuning_full.csv", "V5b_merged.csv"])


# ----------------------------------------------------------------------------
# FIG 11 - Mốc tham chiếu: head-only, full FT, LoRA, AdaLoRA
# ----------------------------------------------------------------------------
def fig11_baselines(ctx):
    head = ctx.csv("results", "reference", "head.csv")
    full = ctx.csv("results", "reference", "full_lr2e5.csv")
    r1 = ctx.csv("results", "global", "global_merged.csv")
    r1 = r1[r1.epochs == 6]
    ada_orig = ctx.csv("results", "adalora", "adalora_results.csv")
    ada_diag = ctx.csv("results", "posthoc", "V6_adalora_diag.csv")
    rows = [("Head only", head, C["grey"], True)]
    rows += [(f"LoRA r={r}", r1[r1["rank"] == r], C["blue"], True) for r in RANKS]
    rows += [("Full fine-tune (LR 2e-5)", full, C["black"], True),
             ("AdaLoRA (orth. reg 0.5)†", ada_orig, C["red"], False),
             ("AdaLoRA (orth. reg off)†", ada_diag, C["red"], False)]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    for ax, y, ttl in [(axes[0], "macro_f1", "Test Macro-F1"), (axes[1], "nll", "Test NLL")]:
        for i, (lab, d, col, fill) in enumerate(rows):
            m, h = mean_ci(d[y].values)
            ax.errorbar(m, i, xerr=h, fmt="o", ms=5, color=col, mfc=col if fill else "white", capsize=2.5, lw=1.3)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] + f" (n={len(r[1])})" for r in rows], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel(ttl)
        ax.grid(axis="y", visible=False)
    axes[0].set_title("F1 (mean ± 95% CI over seeds)", fontsize=8.5)
    axes[1].set_title("NLL", fontsize=8.5)
    fig.text(0.01, -0.02, "† exploratory: LR not tuned for AdaLoRA, 3 seeds; hollow markers; do not read as a method comparison.",
             fontsize=7, color=C["red"])
    fig.tight_layout()
    ctx.save(fig, "fig11_baselines", ["head.csv", "full_lr2e5.csv", "global_merged.csv", "adalora_results.csv", "V6_adalora_diag.csv"])


# ----------------------------------------------------------------------------
# FIG 12 - AdaLoRA: rank cuối theo lớp và loại module
# ----------------------------------------------------------------------------
def fig12_adalora_ranks(ctx):
    files = sorted(glob.glob(ctx.p("results", "adalora", "adalora_rank", "*.json")))
    if not files:
        raise RuntimeError("không có adalora_rank/*.json")
    order = ["query", "key", "value", "attn.out", "ffn.in", "ffn.out"]

    def kind(name):
        if name.endswith("attention.self.query"):
            return "query"
        if name.endswith("attention.self.key"):
            return "key"
        if name.endswith("attention.self.value"):
            return "value"
        if name.endswith("attention.output.dense"):
            return "attn.out"
        if name.endswith("intermediate.dense"):
            return "ffn.in"
        if re.search(r"layer\.\d+\.output\.dense$", name):
            return "ffn.out"
        return None

    mats, vecs, names = [], [], None
    for f in files:
        rk = json.load(open(f, encoding="utf-8"))["ranks"]
        M = np.full((12, len(order)), np.nan)
        keys = []
        for n, v in rk.items():
            m = re.search(r"layer\.(\d+)\.", n)
            k = kind(n)
            if m and k:
                M[int(m.group(1)), order.index(k)] = v
                keys.append(n)
        mats.append(M)
        vecs.append([rk[n] for n in sorted(rk)])
        names = sorted(rk)
    seeds = [re.search(r"__s(\d+)__", os.path.basename(f)).group(1) for f in files]
    taus = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            taus.append(stats.kendalltau(vecs[i], vecs[j])[0])
    vmax = np.nanmax(mats)
    fig, axes = plt.subplots(1, len(mats), figsize=(2.4 * len(mats) + 0.8, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, M, s in zip(axes, mats, seeds):
        im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"seed {s}", fontsize=8.5)
        ax.grid(False)
    axes[0].set_yticks(range(12))
    axes[0].set_ylabel("encoder layer")
    cb = fig.colorbar(im, ax=axes, pad=0.02)
    cb.set_label("final rank")
    tau_txt = ", ".join(f"{t:.2f}" for t in taus)
    fig.suptitle(f"AdaLoRA final rank allocation (exploratory) · Kendall τ between seeds: {tau_txt}", fontsize=8.5, y=1.02)
    ctx.save(fig, "fig12_adalora_rank_allocation", ["results/adalora/adalora_rank/*.json"])


# ----------------------------------------------------------------------------
FIGS = {
    "fig01": (fig01_design, False), "fig02": (fig02_main, False), "fig03": (fig03_forest, False),
    "fig04": (fig04_lr_landscape, False), "fig05": (fig05_h3_reversal, False),
    "fig06": (fig06_calibration, True), "fig07": (fig07_alpha, False), "fig08": (fig08_model_dependence, False),
    "fig09": (fig09_per_class, True), "fig10": (fig10_fp32, False), "fig11": (fig11_baselines, False),
    "fig12": (fig12_adalora_ranks, False),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="thư mục gốc repo (chứa results/)")
    ap.add_argument("--outdir", default=None, help="thư mục đầu ra (mặc định <root>/figures)")
    ap.add_argument("--only", nargs="*", help="ví dụ: fig03 fig05")
    ap.add_argument("--formats", nargs="*", default=["png", "pdf"])
    ap.add_argument("--no-logits", action="store_true", help="bỏ các hình cần logits (.npz)")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    ctx = Ctx(root, a.outdir or os.path.join(root, "figures"), a.formats, not a.no_logits)
    todo = a.only or list(FIGS)
    failed = []
    for key in todo:
        key = key[:5]
        if key not in FIGS:
            print(f"bỏ qua tên không hợp lệ: {key}")
            continue
        fn, needs_logits = FIGS[key]
        if needs_logits and not ctx.use_logits:
            print(f"  [skip] {key} (cần logits)")
            continue
        try:
            fn(ctx)
        except Exception as e:  # tiếp tục các hình khác
            failed.append((key, repr(e)))
            print(f"  [FAIL] {key}: {e!r}")
    if MANIFEST:
        with open(os.path.join(ctx.outdir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(MANIFEST, f, indent=2, ensure_ascii=False)
    if failed:
        print("\nLỗi:", *[f"{k}: {m}" for k, m in failed], sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()

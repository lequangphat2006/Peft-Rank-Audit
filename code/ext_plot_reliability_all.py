"""Reliability + ECE cho 5 rank x 2 encoder, 16 seed moi o.

Fit temperature TREN CALIBRATION split (cal_logits/cal_gold), bao ECE tren
TEST split (eval_logits/eval_gold). Moi seed fit T rieng; bao mean +- SD
tren 16 seed. Reliability diagram ve cho seed co ECE_raw trung vi.

Outputs:
  figures/fig_reliability_all_ranks.png/.pdf
  figures/fig_ece_vs_rank.png/.pdf
  figures/calibration_summary.csv
  figures/calibration_per_seed.csv
"""
import os
import re
import glob
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

RANKS = [1, 4, 8, 16, 32]
MODELS = ["XLM-R", "PhoBERT"]
PATTERN = {
    "XLM-R":   "results/ext/xlmr/logits/T2_XLM__s{seed}__global__L-1__r{rank}__e6__*.npz",
    "PhoBERT": "results/ext/phobert/logits/T2_PHO__s{seed}__global__L-1__r{rank}__e6__*.npz",
}
SEEDS = {
    "XLM-R":   list(range(200, 216)),
    "PhoBERT": list(range(300, 316)),
}
LR_MAP = {
    "XLM-R":   {1: "1e-3", 4: "5e-4", 8: "1e-4", 16: "2e-4", 32: "5e-5"},
    "PhoBERT": {1: "5e-4", 4: "5e-4", 8: "1e-4", 16: "1e-4", 32: "1e-4"},
}
N_BINS = 15
OUT = "figures"
COLORS = {"XLM-R": "#1f77b4", "PhoBERT": "#d95f02"}
LABELS = [0, 1, 2]

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def ece_score(probs, labels, n_bins=N_BINS):
    conf = probs.max(axis=1); pred = probs.argmax(axis=1); ok = (pred == labels)
    edges = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            ece += abs(ok[m].mean() - conf[m].mean()) * m.mean()
    return float(ece)

def reliability(probs, labels, n_bins=N_BINS):
    conf = probs.max(axis=1); pred = probs.argmax(axis=1); ok = (pred == labels)
    edges = np.linspace(0, 1, n_bins + 1); c = 0.5 * (edges[:-1] + edges[1:])
    acc = np.full(n_bins, np.nan); cnt = np.zeros(n_bins, int)
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        cnt[i] = int(m.sum())
        if m.any(): acc[i] = ok[m].mean()
    return c, acc, cnt

def fit_T(logits, labels):
    f = lambda T: log_loss(labels, softmax(logits / T), labels=LABELS)
    return float(minimize_scalar(f, bounds=(0.1, 10.0), method="bounded").x)

def seed_files(model, rank):
    out = []
    for s in SEEDS[model]:
        hits = sorted(glob.glob(PATTERN[model].format(seed=s, rank=rank)))
        if hits:
            out.append(hits[0])
    return out

def main():
    os.makedirs(OUT, exist_ok=True)
    per_seed_rows, summary_rows = [], []

    fig, axes = plt.subplots(2, len(RANKS), figsize=(3.8 * len(RANKS), 7.6))
    fig_ece, ax_ece = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    for mi, model in enumerate(MODELS):
        ece_raw_mean, ece_ts_mean = [], []
        for ri, rank in enumerate(RANKS):
            ax = axes[mi, ri]
            files = seed_files(model, rank)
            if not files:
                ax.text(0.5, 0.5, "missing\n" + model + " r=" + str(rank),
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            seeds = []
            for f in files:
                d = np.load(f)
                if "cal_logits" not in d.files or "cal_gold" not in d.files:
                    raise RuntimeError("Missing cal_* in " + f)
                T = fit_T(d["cal_logits"].astype(np.float64),
                          d["cal_gold"].astype(np.int64))
                p_raw = softmax(d["eval_logits"].astype(np.float64))
                p_ts = softmax(d["eval_logits"].astype(np.float64) / T)
                y = d["eval_gold"].astype(np.int64)
                s = dict(
                    model=model, rank=rank,
                    seed=int(re.search(r"__s(\d+)__", f).group(1)),
                    lr=LR_MAP[model][rank],
                    T=T,
                    acc=float((p_raw.argmax(1) == y).mean()),
                    nll_raw=float(log_loss(y, p_raw, labels=LABELS)),
                    nll_ts=float(log_loss(y, p_ts, labels=LABELS)),
                    ece_raw=ece_score(p_raw, y),
                    ece_ts=ece_score(p_ts, y),
                )
                seeds.append(s)
                per_seed_rows.append(s)

            def agg(k):
                v = np.array([s[k] for s in seeds])
                return float(v.mean()), float(v.std(ddof=1))

            row = dict(model=model, rank=rank, n_seeds=len(seeds),
                       lr=LR_MAP[model][rank])
            for k in ("T", "acc", "nll_raw", "nll_ts", "ece_raw", "ece_ts"):
                m, sd = agg(k)
                row[k + "_mean"], row[k + "_sd"] = m, sd
            summary_rows.append(row)

            # reliability diagram: seed có ECE_raw trung vị
            med = sorted(seeds, key=lambda s: s["ece_raw"])[len(seeds) // 2]
            d = np.load(seed_files(model, rank)[seeds.index(med)])
            p_raw = softmax(d["eval_logits"].astype(np.float64))
            p_ts = softmax(d["eval_logits"].astype(np.float64) / med["T"])
            y = d["eval_gold"].astype(np.int64)
            c, a_raw, _ = reliability(p_raw, y)
            _, a_ts, _ = reliability(p_ts, y)

            w = 1.0 / N_BINS
            ax.bar(c - w / 4, np.nan_to_num(a_raw), width=w / 2,
                   color="0.72", edgecolor="k", linewidth=0.4, label="raw")
            ax.bar(c + w / 4, np.nan_to_num(a_ts), width=w / 2,
                   color="0.25", edgecolor="k", linewidth=0.4, label="TS")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(model + "  rank=" + str(rank) +
                         "\nECE " + format(row["ece_raw_mean"], ".3f") +
                         " -> " + format(row["ece_ts_mean"], ".3f") +
                         "  (median seed " + str(med["seed"]) + ")",
                         fontsize=8.5)
            ax.set_xlabel("confidence", fontsize=8)
            if ri == 0:
                ax.set_ylabel("accuracy", fontsize=8)
                ax.legend(loc="upper left", fontsize=7)

            ece_raw_mean.append(row["ece_raw_mean"])
            ece_ts_mean.append(row["ece_ts_mean"])

            print(model, "r=" + str(rank),
                  "| n=" + str(len(seeds)),
                  "| ECE raw " + format(row["ece_raw_mean"], ".4f") +
                  " +- " + format(row["ece_raw_sd"], ".4f"),
                  "| T " + format(row["T_mean"], ".2f") +
                  " +- " + format(row["T_sd"], ".2f"),
                  "| ECE TS " + format(row["ece_ts_mean"], ".4f") +
                  " +- " + format(row["ece_ts_sd"], ".4f"),
                  "| NLL raw " + format(row["nll_raw_mean"], ".4f") +
                  " | NLL TS " + format(row["nll_ts_mean"], ".4f"))

        ax2 = ax_ece[mi]
        ax2.plot(RANKS, ece_raw_mean, "o--", color=COLORS[model],
                 label="raw", alpha=0.55)
        ax2.plot(RANKS, ece_ts_mean, "o-", color=COLORS[model],
                 label="temperature scaled")
        ax2.set_xscale("log", base=2)
        ax2.set_xticks(RANKS); ax2.set_xticklabels([str(r) for r in RANKS])
        ax2.set_xlabel("global LoRA rank")
        ax2.set_title(model + "  (mean over 16 seeds)")
        ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    ax_ece[0].set_ylabel("ECE (15 bins)")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_reliability_all_ranks.png"), dpi=300)
    fig.savefig(os.path.join(OUT, "fig_reliability_all_ranks.pdf"))
    fig_ece.tight_layout()
    fig_ece.savefig(os.path.join(OUT, "fig_ece_vs_rank.png"), dpi=300)
    fig_ece.savefig(os.path.join(OUT, "fig_ece_vs_rank.pdf"))

    def dump(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); [w.writerow(r) for r in rows]

    dump(os.path.join(OUT, "calibration_summary.csv"), summary_rows)
    dump(os.path.join(OUT, "calibration_per_seed.csv"), per_seed_rows)

    print("\nSaved figures/fig_reliability_all_ranks.png/.pdf")
    print("       figures/fig_ece_vs_rank.png/.pdf")
    print("       figures/calibration_summary.csv")
    print("       figures/calibration_per_seed.csv")

if __name__ == "__main__":
    main()

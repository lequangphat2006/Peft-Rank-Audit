"""ext_common.py - hàm dùng chung cho phần mở rộng EXT-1 (PhoBERT 5 rank x 16 seed).

Không huấn luyện gì ở đây. Chỉ gồm:
  * đọc CSV kết quả, lọc nhánh LR (LR theo rank / LR cố định / LR tốt nhì),
  * ước lượng hiệu ứng rank theo từng seed (độ dốc theo log2(rank)),
  * Welch, TOST ghép cặp, hiệu chỉnh Holm,
  * Macro-F1 nhanh bằng bincount (dùng cho bootstrap hai tầng).

Quy ước giữ nguyên như code/analyze_global.py: estimand "tổng hiệu ứng" = độ dốc theo
log2(rank) x khoảng log2 (rank 1 -> 32 = 5 đơn vị log2). Với thiết kế cân bằng
(mọi seed có đủ mọi rank) độ dốc OLS có hiệu ứng cố định theo seed bằng đúng trung bình
các độ dốc theo từng seed, nên hai cách tính cho cùng một số.
"""
import os
import sys
import json
import hashlib

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CRASH_F1 = 0.25          # định nghĩa "lượt sập" (docs/analysis_plan_v2.0.md mục 9, 2026-10-17)
ALL_RANKS = [1, 4, 8, 16, 32]


# ------------------------------------------------------------------ I/O
def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_lr_map(path):
    """Đọc {rank: lr} hoặc {rank: {"best_lr": lr, ...}} -> {int: float}."""
    raw = json.load(open(path, encoding="utf-8-sig"))
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            v = v.get("best_lr")
        if v is not None:
            out[int(k)] = float(v)
    return out


def load_second_best(path):
    raw = json.load(open(path, encoding="utf-8-sig"))
    return {int(k): float(v["second_best_lr"]) for k, v in raw.items()
            if isinstance(v, dict) and v.get("second_best_lr") is not None}


# ------------------------------------------------------------ chọn nhánh LR
def select_branch(df, lr_for_rank):
    """Giữ các dòng có lr == lr_for_rank[rank]. Rank không có trong map bị loại."""
    keep = np.zeros(len(df), dtype=bool)
    for r, lr in lr_for_rank.items():
        keep |= ((df["rank"] == r) & np.isclose(df["lr"], lr)).to_numpy()
    return df[keep].copy()


def select_fixed_lr(df, lr):
    return df[np.isclose(df["lr"], lr)].copy()


def mark_crash(df, col="macro_f1", thr=CRASH_F1):
    d = df.copy()
    d["crash"] = d[col].astype(float) < thr
    return d


# ------------------------------------------------------ hiệu ứng theo seed
def per_seed_effects(d, y="macro_f1", ranks=None):
    """Mỗi seed -> độ dốc theo log2(rank), tổng hiệu ứng (= dốc x khoảng log2) và hiệu rank cao nhất - thấp nhất.

    Yêu cầu đúng 1 dòng cho mỗi (seed, rank). Seed thiếu rank bị loại và được trả về trong `dropped`.
    """
    ranks = sorted(ranks if ranks is not None else d["rank"].unique())
    d = d[d["rank"].isin(ranks)]
    dup = d.groupby(["seed", "rank"]).size()
    if (dup > 1).any():
        bad = dup[dup > 1].index.tolist()[:5]
        raise ValueError(f"nhiều hơn 1 dòng cho cùng (seed, rank), ví dụ {bad}; hãy lọc theo nhánh LR trước")
    rows, dropped = [], []
    x = np.log2(np.asarray(ranks, dtype=float))
    span = float(x.max() - x.min())
    for seed, g in d.groupby("seed"):
        g = g.set_index("rank")
        if not all(r in g.index for r in ranks):
            dropped.append(int(seed))
            continue
        yv = g.loc[ranks, y].to_numpy(dtype=float)
        b = float(np.polyfit(x, yv, 1)[0])
        rows.append(dict(seed=int(seed), slope=b, total=b * span, diff=float(yv[-1] - yv[0])))
    return pd.DataFrame(rows), dropped


def welch(a, b):
    """Hiệu a - b, Welch t, df Satterthwaite, CI 95%, p hai phía."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    diff = float(a.mean() - b.mean())
    df = float((va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)))
    t = diff / se
    q = stats.t.ppf(0.975, df)
    return dict(diff=diff, se=se, df=df, t=float(t), p=float(2 * (1 - stats.t.cdf(abs(t), df))),
                ci95=[diff - q * se, diff + q * se], n_a=len(a), n_b=len(b))


def perm_p(a, b, n_perm=20000, seed=0):
    """Kiểm định hoán vị hai phía cho hiệu trung bình (kiểm tra chéo của Welch)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    pool, na = np.concatenate([a, b]), len(a)
    obs = abs(a.mean() - b.mean())
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        cnt += abs(pool[:na].mean() - pool[na:].mean()) >= obs - 1e-15
    return float((cnt + 1) / (n_perm + 1))


def one_sample(x):
    x = np.asarray(x, float)
    n = len(x)
    m, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(n))
    return m, se, n - 1


def tost_from(m, se, df, delta):
    """TOST hai phía một mẫu; trả (p_tost, CI90)."""
    t_lo, t_hi = (m + delta) / se, (delta - m) / se
    p = float(max(1 - stats.t.cdf(t_lo, df), 1 - stats.t.cdf(t_hi, df)))
    q = stats.t.ppf(0.95, df)
    return p, [m - q * se, m + q * se]


def onesided_pos_p(m, se, df):
    return float(1 - stats.t.cdf(m / se, df))


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m, adj, run = len(ps), np.empty(len(ps)), 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


# --------------------------------------------------------------- F1 nhanh
def macro_f1_fast(gold, pred, K=3):
    """Macro-F1 bằng bincount; gold, pred là mảng int cùng độ dài. Lớp vắng mặt có F1 = 0 (như zero_division=0)."""
    cm = np.bincount(gold * K + pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(K), where=denom > 0)
    return float(f1.mean())


def log_summary(msg):
    print(msg, flush=True)


try:  # Windows cp1252 -> UTF-8 (giống các script cũ trong repo)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

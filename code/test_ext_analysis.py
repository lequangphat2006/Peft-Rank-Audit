"""Kiểm thử tự động cho các script EXT-1 bằng dữ liệu tổng hợp (không cần GPU, không cần dữ liệu thật).

  python -m pytest code/test_ext_analysis.py -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ext_common as C  # noqa: E402
import ext_lr_rule as L  # noqa: E402
import ext_merge as M  # noqa: E402
import ext_confirmatory as X  # noqa: E402
import ext_preflight as P  # noqa: E402


# ------------------------------------------------------------------ helpers
def tune_df(table, seeds=(400, 401, 402)):
    """table: {rank: {lr: f1}} -> DataFrame giống CSV tune của run_pilot.py."""
    rows = []
    for r, d in table.items():
        for lr, f1 in d.items():
            for s in seeds:
                rows.append(dict(config="global", rank=r, lr=lr, seed=s, macro_f1=f1 + 0.001 * (s - 401)))
    return pd.DataFrame(rows)


def runs_df(model, seeds, ranks, f1_of, lr=2e-4, cell="B1"):
    rows = []
    for s in seeds:
        for r in ranks:
            f1 = f1_of(s, r)
            rows.append(dict(run_id=f"{cell}__s{s}__global__L-1__r{r}__e6__lr{lr:g}", cell=cell, seed=s, rank=r,
                             epochs=6, lr=lr, macro_f1=f1, nll=0.2 + 0.001 * np.log2(r), f1_neu=f1 - 0.2))
    return pd.DataFrame(rows)


# -------------------------------------------------------------- quy tắc LR
def test_lr_rule_flags_low_edge_and_extends():
    base = {32: {1e-4: 0.817, 2e-4: 0.806, 5e-4: 0.62, 1e-3: 0.25}}
    dec, needs = L.decide(tune_df(base), stage="base")
    assert dec[32]["best_lr"] == 1e-4 and dec[32]["edge_of_base_grid"] == "low"
    assert needs == {32: "low"}
    ext = dict(base[32]); ext[5e-5] = 0.830
    dec, _ = L.decide(tune_df({32: ext}), stage="final")
    assert dec[32]["best_lr"] == 5e-5 and dec[32]["extended"]
    assert dec[32]["still_on_edge_after_extension"] == "low"   # không mở rộng thêm một bước nữa


def test_lr_rule_extension_not_better_keeps_base_choice():
    t = {1: {1e-4: 0.79, 2e-4: 0.81, 5e-4: 0.805, 1e-3: 0.816}}
    dec, needs = L.decide(tune_df(t), stage="base")
    assert needs == {1: "high"}
    t[1][2e-3] = 0.80                                  # đã chạy mở rộng nhưng không tốt hơn
    dec, needs = L.decide(tune_df(t), stage="final")
    assert dec[1]["best_lr"] == 1e-3 and dec[1]["still_on_edge_after_extension"] is None


def test_lr_rule_tie_goes_to_smaller_lr_and_ignores_unneeded_ext():
    t = {8: {1e-4: 0.8143, 2e-4: 0.8119, 5e-4: 0.8176, 1e-3: 0.8064, 5e-5: 0.9}}
    dec, _ = L.decide(tune_df(t), stage="final")
    # 1e-4 và 5e-4 hòa (< 0,005) -> chọn nhỏ hơn = 1e-4 = biên dưới -> được xét 5e-5
    assert dec[8]["edge_of_base_grid"] == "low"
    assert dec[8]["best_lr"] == 5e-5


def test_lr_rule_crash_rate_reported():
    t = tune_df({16: {1e-4: 0.8, 2e-4: 0.82, 5e-4: 0.8, 1e-3: 0.22}})
    dec, _ = L.decide(t, stage="final")
    row = [x for x in dec[16]["table"] if np.isclose(x["lr"], 1e-3)][0]
    assert row["crash_rate"] == 1.0


# -------------------------------------------------------- thống kê chung
def test_per_seed_effects_recovers_slope():
    d = runs_df("P", range(300, 306), [1, 4, 8, 16, 32], lambda s, r: 0.80 + 0.004 * np.log2(r))
    e, drop = C.per_seed_effects(d, "macro_f1", [1, 4, 8, 16, 32])
    assert drop == [] and np.allclose(e["total"], 0.02)
    assert np.allclose(e["diff"], 0.02)


def test_per_seed_effects_drops_incomplete_and_rejects_duplicates():
    d = runs_df("P", [300, 301], [1, 4, 8], lambda s, r: 0.8)
    e, drop = C.per_seed_effects(d[~((d.seed == 301) & (d["rank"] == 8))], "macro_f1", [1, 4, 8])
    assert drop == [301] and len(e) == 1
    with pytest.raises(ValueError):
        C.per_seed_effects(pd.concat([d, d.iloc[:1]]), "macro_f1", [1, 4, 8])


def test_holm_matches_known_values():
    assert np.allclose(C.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_macro_f1_fast_matches_sklearn():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 3, 500); p = rng.integers(0, 3, 500)
    assert abs(C.macro_f1_fast(g, p) - f1_score(g, p, average="macro")) < 1e-12
    assert abs(C.macro_f1_fast(g, np.zeros_like(g)) - f1_score(g, np.zeros_like(g), average="macro", zero_division=0)) < 1e-12


def test_e1_detects_interaction_and_null():
    rng = np.random.default_rng(1)
    ranks = [1, 4, 8, 16, 32]
    p = runs_df("P", range(300, 316), ranks, lambda s, r: 0.80 + 0.004 * np.log2(r) + rng.normal(0, .004))
    x_null = runs_df("X", range(200, 216), ranks, lambda s, r: 0.84 + rng.normal(0, .004), cell="R1")
    w, _, _ = X.e1_from(p, x_null, ranks)
    assert w["p"] < 0.01 and 0.01 < w["diff"] < 0.03
    x_same = runs_df("X", range(200, 216), ranks, lambda s, r: 0.84 + 0.004 * np.log2(r) + rng.normal(0, .004), cell="R1")
    w2, _, _ = X.e1_from(p, x_same, ranks)
    assert w2["p"] > 0.05 and w2["ci95"][0] < 0 < w2["ci95"][1]


# --------------------------------------------------------------- gộp shard
def test_merge_dedupe_conflict_and_missing(tmp_path):
    d = runs_df("P", [300, 301], [1, 8], lambda s, r: 0.8)
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    d.to_csv(a, index=False)
    d2 = d.copy(); d2.loc[0, "macro_f1"] = 0.9       # cùng run_id, số liệu khác
    d2.to_csv(b, index=False)
    df, src, n_dup, conflicts = M.merge([str(a), str(b)])
    assert len(df) == 4 and n_dup == 4 and len(conflicts) == 1
    assert M.parse_seeds(["300-302", "400"]) == [300, 301, 302, 400]


# --------------------------------------------------------- bootstrap hai tầng
def make_logits(dirpath, df, acc_of, N=400, seed=0):
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, 3, N)
    os.makedirs(dirpath, exist_ok=True)
    for _, r in df.iterrows():
        pr = np.where(rng.random(N) < acc_of(r["rank"]), gold, (gold + 1 + rng.integers(0, 2, N)) % 3)
        lg = np.eye(3)[pr] * 3 + rng.normal(0, .1, (N, 3))
        np.savez_compressed(os.path.join(dirpath, r["run_id"] + ".npz"), eval_logits=lg, eval_gold=gold)


def test_two_level_bootstrap_separates_effect_from_null(tmp_path):
    ranks = [1, 8, 32]
    p = runs_df("P", range(300, 308), ranks, lambda s, r: 0.0, cell="B1")
    x = runs_df("X", range(200, 208), ranks, lambda s, r: 0.0, cell="R1")
    make_logits(tmp_path / "p", p, lambda r: 0.80 + 0.03 * np.log2(r) / 5, seed=3)
    make_logits(tmp_path / "x", x, lambda r: 0.80, seed=3)    # cùng seed -> cùng gold
    b = X.two_level_bootstrap(p, x, str(tmp_path / "p"), str(tmp_path / "x"), ranks, n_boot=150)
    assert b["point"]["delta"] > 0.01
    assert b["two_level"]["ci95_delta"][1] - b["two_level"]["ci95_delta"][0] >= \
        b["seed_only"]["ci95_delta"][1] - b["seed_only"]["ci95_delta"][0] - 1e-9


def test_bootstrap_rejects_different_gold(tmp_path):
    ranks = [1, 32]
    p = runs_df("P", [300, 301], ranks, lambda s, r: 0.0, cell="B1")
    x = runs_df("X", [200, 201], ranks, lambda s, r: 0.0, cell="R1")
    make_logits(tmp_path / "p", p, lambda r: 0.8, seed=1)
    make_logits(tmp_path / "x", x, lambda r: 0.8, seed=2)    # gold khác
    with pytest.raises(ValueError):
        X.two_level_bootstrap(p, x, str(tmp_path / "p"), str(tmp_path / "x"), ranks, n_boot=5)


# ----------------------------------------------------------------- preflight
def test_preflight_detects_missing_rank_and_seed_overlap(tmp_path, capsys):
    import json
    f = tmp_path / "m.json"
    f.write_text(json.dumps({"1": 2e-4, "4": 2e-4, "16": 2e-4, "32": 1e-4}))   # thiếu rank 8 như phobert_lr_map.json cũ
    P._fail = 0
    P.check_lr_map(str(f), [1, 4, 8, 16, 32])
    assert P._fail == 1
    P._fail = 0
    P.check_seeds(list(range(300, 316)), [300, 301, 302])
    assert P._fail == 1


def test_alpha_paired_difference():
    import ext_alpha as A
    rng = np.random.default_rng(5)
    rows = []
    for s in range(300, 308):
        for r in (1, 8, 32):
            for mode, slope in (("2r", 0.004), ("fixed:16", -0.004)):
                rows.append(dict(run_id=f"A__{s}__{r}__{mode}", seed=s, rank=r, epochs=6, lr=2e-4, alpha_mode=mode,
                                 macro_f1=0.8, nll=0.2 + slope * np.log2(r) / 5 + rng.normal(0, 1e-4)))
    out = A.analyse(pd.DataFrame(rows), [1, 8, 32], 2e-4)
    d = out["paired_nll_fixed16_minus_2r"]
    assert d["est"] < -0.003 and d["ci95"][1] < 0 and d["n"] == 8

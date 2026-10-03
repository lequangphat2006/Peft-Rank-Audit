"""Core utilities for the capacity-need audit (Paper A).

Unit of analysis m = transformer layer (12 layers). One rank value is applied to all
6 LoRA target modules inside a layer. Module names are listed explicitly (never the
suffix "dense"), so the classification head is never adapted by accident.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

MODULE_SUFFIXES = [
    "attention.self.query",
    "attention.self.key",
    "attention.self.value",
    "attention.output.dense",
    "intermediate.dense",
    "output.dense",
]
# module "types" for the type-level sweep (C7): rank applied to the same type in ALL layers
TYPE_GROUPS = {s: [s] for s in MODULE_SUFFIXES}


def encoder_prefix(model) -> str:
    """Return e.g. 'roberta.encoder.layer' for Roberta/XLM-R/PhoBERT, 'bert.encoder.layer' for BERT."""
    for name, _ in model.named_modules():
        if name.endswith("encoder.layer.0"):
            return name[: -len(".0")]
    raise ValueError("could not locate encoder.layer.0 in model")


def n_layers(model) -> int:
    prefix = encoder_prefix(model)
    idx = {int(n[len(prefix) + 1 :].split(".")[0]) for n, _ in model.named_modules()
           if n.startswith(prefix + ".") and n[len(prefix) + 1 :].split(".")[0].isdigit()}
    return max(idx) + 1


def module_names(model, layers: Sequence[int] | None = None, suffixes: Sequence[str] = MODULE_SUFFIXES) -> List[str]:
    prefix = encoder_prefix(model)
    layers = range(n_layers(model)) if layers is None else layers
    return [f"{prefix}.{i}.{s}" for i in layers for s in suffixes]


def build_patterns(model, layer_ranks: Dict[int, int] | Sequence[int], base_r: int = 8,
                   type_ranks: Dict[str, int] | None = None, alpha_ratio: float = 2.0):
    """Return (target_modules, rank_pattern, alpha_pattern).

    layer_ranks: rank for each layer (all 6 modules). type_ranks (optional) overrides the
    rank of one module type in every layer (type-level sweep).
    alpha = alpha_ratio * r for every module, so the LoRA scaling alpha/r stays constant.
    """
    prefix = encoder_prefix(model)
    L = n_layers(model)
    if not isinstance(layer_ranks, dict):
        layer_ranks = {i: int(r) for i, r in enumerate(layer_ranks)}
    targets, rank_pattern, alpha_pattern = [], {}, {}
    for i in range(L):
        for s in MODULE_SUFFIXES:
            name = f"{prefix}.{i}.{s}"
            r = int(layer_ranks.get(i, base_r))
            if type_ranks and s in type_ranks:
                r = int(type_ranks[s])
            targets.append(name)
            rank_pattern[name] = r
            alpha_pattern[name] = int(round(alpha_ratio * r))
    return targets, rank_pattern, alpha_pattern


def lora_config(model, layer_ranks, base_r=8, type_ranks=None, dropout=0.1, alpha_ratio=2.0):
    from peft import LoraConfig, TaskType
    targets, rank_pattern, alpha_pattern = build_patterns(model, layer_ranks, base_r, type_ranks, alpha_ratio)
    return LoraConfig(task_type=TaskType.SEQ_CLS, r=base_r, lora_alpha=int(round(alpha_ratio * base_r)),
                      lora_dropout=dropout, target_modules=targets, rank_pattern=rank_pattern,
                      alpha_pattern=alpha_pattern, bias="none")


def count_lora_params(peft_model) -> int:
    return sum(p.numel() for n, p in peft_model.named_parameters() if "lora_" in n)


# --------------------------------------------------------------------------------------
# Budget-matched rank allocation (layer level)
# --------------------------------------------------------------------------------------
def allocate_ranks(sigma: Sequence[float], total: int, r_min: int = 1, r_max: int = 32) -> np.ndarray:
    """Map scores sigma in [0,1] to integer ranks with sum == total, r_min <= r <= r_max.

    r_float = clip(a * (r_min + (r_max - r_min) * sigma), r_min, r_max) with the scale a chosen
    so that sum(r_float) == total (scale first), then floor + largest remainder (round last).
    At layer level every rank unit costs the same number of parameters in every layer, so
    matching the rank sum also matches the parameter count (checked in test_rankcap.py).
    """
    s = np.asarray(sigma, dtype=float)
    n = len(s)
    if not (n * r_min <= total <= n * r_max):
        raise ValueError("total not reachable within [r_min, r_max]")
    base = r_min + (r_max - r_min) * s

    def f(a):
        return np.clip(a * base, r_min, r_max).sum()

    lo, hi = 0.0, 1.0
    while f(hi) < total and hi < 1e9:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) < total:
            lo = mid
        else:
            hi = mid
    r_float = np.clip(hi * base, r_min, r_max)
    r_floor = np.floor(r_float + 1e-12).astype(int)
    r_floor = np.clip(r_floor, r_min, r_max)
    remaining = int(total - r_floor.sum())
    frac = r_float - r_floor
    order = np.argsort(-frac, kind="stable")
    r = r_floor.copy()
    if remaining > 0:
        for i in order:
            if remaining == 0:
                break
            if r[i] < r_max:
                r[i] += 1
                remaining -= 1
    elif remaining < 0:
        for i in order[::-1]:
            if remaining == 0:
                break
            if r[i] > r_min:
                r[i] -= 1
                remaining += 1
    assert r.sum() == total and r.min() >= r_min and r.max() <= r_max
    return r


def normalize01(x: Sequence[float]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min()
    return np.full_like(x, 0.5) if rng == 0 else (x - x.min()) / rng


def make_controls(sigma: Sequence[float], total: int, rng: np.random.Generator,
                  r_min: int = 1, r_max: int = 32) -> Dict[str, np.ndarray]:
    sigma = normalize01(sigma)
    n = len(sigma)
    return {
        "uniform": np.full(n, total // n, dtype=int),
        "dynamic": allocate_ranks(sigma, total, r_min, r_max),
        "random": allocate_ranks(rng.uniform(0, 1, n), total, r_min, r_max),
        "reversed": allocate_ranks(1 - sigma, total, r_min, r_max),
        "permuted": allocate_ranks(rng.permutation(sigma), total, r_min, r_max),
    }


# --------------------------------------------------------------------------------------
# Class-imbalance protocol (C6)
# --------------------------------------------------------------------------------------
def imbalanced_counts(K: int, N: int, rho: float) -> np.ndarray:
    """n_k proportional to rho^(-k/(K-1)), sum == N (largest remainder)."""
    w = np.array([rho ** (-k / (K - 1)) for k in range(K)]) if K > 1 else np.ones(1)
    raw = N * w / w.sum()
    n = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - n))[: N - n.sum()]:
        n[i] += 1
    return n


def max_feasible_N(class_counts: Sequence[int]) -> int:
    """Largest N supporting rho = 1 (all classes equal): K * smallest class count."""
    return len(class_counts) * int(min(class_counts))


def feasible_imbalance(class_counts: Sequence[int], N: int, rhos=(1, 5, 20)) -> Dict[float, bool]:
    """Check each rho: do the needed per-class counts fit inside the available classes?
    Classes are assigned largest-need-first to the largest available class."""
    avail = sorted(class_counts, reverse=True)
    out = {}
    for rho in rhos:
        need = sorted(imbalanced_counts(len(class_counts), N, rho), reverse=True)
        out[rho] = all(nd <= av for nd, av in zip(need, avail))
    return out

"""run_pilot.py v3 - train one LoRA configuration per job and append its metrics to a CSV.

What is new in v3 (everything else behaves like v2)
  * --method {lora,full,head}  : LoRA (default), full fine-tuning, or classifier-head-only (floor).
  * --alpha_mode {2r,fixed:N,rslora} : alpha = 2r (default), alpha = N constant, or rsLoRA scaling.
  * --data_fraction F          : stratified subsample of the training set (F in (0,1]); F goes into run_id.
  * delta-W statistics (effective rank, stable rank, Frobenius norm) computed BEFORE del model,
    written to deltaw/<run_id>.json. Uses QR of B and A^T so no d x k matrix is ever materialised.
  * resolve_precision: 'auto' now picks fp16 on GPUs without native bf16 (compute capability < 8.0);
    the old v2 behaviour (bf16 emulated on T4) was a PyTorch is_bf16_supported() false positive.
  * New CSV columns: method, alpha_mode, data_fraction, n_train.

What was already in v2
  * --sweep_global, --epochs_list / --lr_list, --shard i/n, dual eval (cal + primary),
    per-epoch epochwise/<run_id>.json, logits/<run_id>.npz, run_id resume, meta.jsonl,
    --class_weight {none,balanced}, --precision {auto,bf16,fp16,fp32}, --dry_run.

Examples
  # Phase 1a, GPU 0 of 2, 16 seeds split into 2 batches of 8
  CUDA_VISIBLE_DEVICES=0 python run_pilot.py --data_dir data/vsfc_sent --out results/global_gpu0.csv \
      --cell R1 --sweep_global --ranks 1 4 8 16 32 --epochs_list 3 6 \
      --seeds 200 201 202 203 204 205 206 207 --shard 0/2 --precision fp16

  # LR tuning on calibration only (3 seeds x 5 ranks x 4 LRs, 6 epochs)
  python run_pilot.py --data_dir data/vsfc_sent --out results/tuning.csv --cell TUNE \
      --sweep_global --ranks 1 4 8 16 32 --lr_list 1e-4 2e-4 5e-4 1e-3 --epochs_list 6 \
      --seeds 400 401 402 --precision fp16 --eval_file calibration.csv

  # Floor / ceiling reference
  python run_pilot.py ... --method head  --seeds 200 201 202 203 204 205 206 207 --epochs 6
  python run_pilot.py ... --method full  --seeds 200 201 202 203 204 --epochs 6

  # alpha sensitivity (Phase C)
  python run_pilot.py ... --alpha_mode fixed:16 --ranks 1 8 32 --seeds 200 201 202 203 204
  python run_pilot.py ... --alpha_mode rslora  --ranks 1 8 32 --seeds 200 201 202 203 204

  # data fraction (Phase 3)
  python run_pilot.py ... --data_fraction 0.2 --ranks 1 8 32 --seeds 200 201 202 203 204 205 206 207
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse, csv, json, random, subprocess, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

import rankcap

GROUPS = {"bottom": [0, 1, 2, 3], "middle": [4, 5, 6, 7], "top": [8, 9, 10, 11]}
MODULE_TYPES = list(rankcap.MODULE_SUFFIXES)

CSV_COLUMNS = [
    "run_id", "cell", "seed", "config", "layer", "rank", "epochs", "lr", "batch_size", "max_len",
    "precision", "class_weight", "limit_train", "method", "alpha_mode", "data_fraction", "n_train",
    "macro_f1", "f1_neg", "f1_neu", "f1_pos", "f1_per_class", "accuracy", "nll",
    "cal_macro_f1", "cal_accuracy", "cal_nll",
    "lora_params", "seconds", "eval_file", "git",
]


# ----------------------------------------------------------------------------- utilities
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [int(r["label"]) for r in rows]


def stratified_subsample(texts, labels, fraction, seed):
    """Keep round(fraction * n_k) rows per class, seed-dependent. Always keeps at least 1 per class."""
    if fraction >= 1.0:
        return texts, labels
    rng = np.random.default_rng(seed)
    y = np.asarray(labels)
    keep = np.zeros(len(y), dtype=bool)
    for k in np.unique(y):
        idx = np.where(y == k)[0]
        n_k = max(1, int(round(fraction * len(idx))))
        keep[rng.choice(idx, size=n_k, replace=False)] = True
    sel = np.where(keep)[0]
    return [texts[i] for i in sel], [int(y[i]) for i in sel]


def tokenize_split(tok, texts, labels, max_len):
    enc = tok(texts, truncation=True, max_length=max_len, padding=False)
    return [{"input_ids": ids, "attention_mask": am, "labels": y}
            for ids, am, y in zip(enc["input_ids"], enc["attention_mask"], labels)]


def make_loader(tok, items, bs, shuffle, seed):
    def collate(batch):
        ys = torch.tensor([b["labels"] for b in batch])
        pad = tok.pad([{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch],
                      padding=True, return_tensors="pt")
        return pad["input_ids"], pad["attention_mask"], ys
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(items, batch_size=bs, shuffle=shuffle, generator=g if shuffle else None,
                      collate_fn=collate)


def resolve_precision(device, choice):
    """Return (name, autocast dtype or None).

    v3 change: 'auto' picks fp16 on GPUs without NATIVE bf16 (compute capability < 8.0, e.g. T4/Turing).
    Reason: torch.cuda.is_bf16_supported() returns True on T4 (it only checks the CUDA version), so v2
    silently ran EMULATED bf16 and lost ~1.5-2x throughput. Emulated bf16 is correct numerically, so
    existing v2 results are still valid; we simply do not want to keep paying the cost in Phase 1.
    Never mix precisions inside one comparison (see docs/analysis_plan_v2.md section 2.1).
    """
    if device != "cuda":
        return ("bf16", None) if choice == "auto" else (choice, None)
    cap_major = torch.cuda.get_device_capability(0)[0]
    native_bf16 = cap_major >= 8
    if choice == "auto":
        choice = "bf16" if native_bf16 else "fp16"
    if choice == "bf16":
        if not native_bf16:
            print("WARNING: bf16 requested on a GPU without native bf16 (compute capability "
                  f"{torch.cuda.get_device_capability(0)}). It will be EMULATED and slow. "
                  "Prefer --precision fp16 for Phase 1 (never mix precisions inside one comparison).",
                  flush=True)
        return "bf16", torch.bfloat16
    if choice == "fp16":
        return "fp16", torch.float16
    return "fp32", None


def git_hash():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                                    stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=here,
                                        stderr=subprocess.DEVNULL).decode().strip()
        return h + ("+dirty" if dirty else "")
    except Exception:
        return "nogit"


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, dl, device, amp_dtype, K):
    model.eval()
    logits, gold = [], []
    for ids, mask, y in dl:
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        logits.append(out.float().cpu()); gold += y.tolist()
    logits = torch.cat(logits).numpy()
    pred = logits.argmax(-1)
    labels = list(range(K))
    per_class = f1_score(gold, pred, average=None, labels=labels, zero_division=0)
    nll = F.cross_entropy(torch.from_numpy(logits).double(), torch.tensor(gold)).item()
    return {"macro_f1": float(f1_score(gold, pred, average="macro", labels=labels, zero_division=0)),
            "per_class": [float(x) for x in per_class],
            "accuracy": float(accuracy_score(gold, pred)), "nll": float(nll)}, logits, np.asarray(gold)


def train_and_eval(model, train_dl, cal_dl, eval_dl, epochs, lr, device, amp_dtype, K,
                   class_weights=None, epoch_eval="cal", warmup=0.1, wd=0.01, clip=1.0):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8)
    total = epochs * len(train_dl); warm = max(1, int(warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    model.to(device)
    w = class_weights.to(device) if class_weights is not None else None

    epochwise, final = [], {}
    for ep in range(1, epochs + 1):
        model.train()
        run_loss, nb = 0.0, 0
        for ids, mask, y in train_dl:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = F.cross_entropy(logits.float(), y, weight=w)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, clip)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
            run_loss += float(loss.detach()); nb += 1
        rec = {"epoch": ep, "train_loss": run_loss / max(1, nb), "lr": float(opt.param_groups[0]["lr"])}
        last = ep == epochs
        if cal_dl is not None and (epoch_eval in ("cal", "both") or last):
            m, lg, gd = evaluate(model, cal_dl, device, amp_dtype, K)
            rec.update({"cal_macro_f1": m["macro_f1"], "cal_nll": m["nll"], "cal_accuracy": m["accuracy"]})
            if last: final["cal"] = (m, lg, gd)
        if eval_dl is not None and (epoch_eval == "both" or last):
            m, lg, gd = evaluate(model, eval_dl, device, amp_dtype, K)
            rec.update({"eval_macro_f1": m["macro_f1"], "eval_nll": m["nll"], "eval_accuracy": m["accuracy"]})
            if last: final["eval"] = (m, lg, gd)
        epochwise.append(rec)
    return final, epochwise


# ------------------------------------------------------------------- delta-W statistics
def delta_w_stats(peft_model):
    """Per-LoRA-layer statistics of delta_W = B @ A.

    Computed with QR of B (d x r) and A^T (k x r) so only an r x r matrix is factorised:
        B = Qb Rb ,  A = Ra^T Qa^T  =>  delta_W = Qb (Rb Ra^T) Qa^T
        singular values of delta_W == singular values of Rb @ Ra^T.
    Returns {module_name: {effective_rank, stable_rank, fro_norm, rank, singular_values}}.
    """
    try:
        from peft.tuners.lora import Linear as LoraLinear
    except Exception:
        try:
            from peft.tuners.lora.layer import Linear as LoraLinear
        except Exception:
            return {}
    stats = {}
    for name, mod in peft_model.named_modules():
        if not isinstance(mod, LoraLinear):
            continue
        try:
            A = mod.lora_A["default"].weight.detach().float().cpu()   # (r, k)
            B = mod.lora_B["default"].weight.detach().float().cpu()   # (d, r)
        except (KeyError, AttributeError):
            continue
        if B.numel() == 0 or A.numel() == 0:
            continue
        _, Rb = torch.linalg.qr(B)        # Rb: (r, r)
        _, Ra = torch.linalg.qr(A.T)      # Ra: (r, r)
        M = Rb @ Ra.T                     # (r, r)
        s = torch.linalg.svdvals(M).clamp_min(1e-12)
        p = s / s.sum()
        stats[name] = {
            "effective_rank": float(torch.exp(-(p * torch.log(p)).sum())),
            "stable_rank": float((s.sum() ** 2) / (s.max() ** 2)),
            "fro_norm": float((s ** 2).sum().sqrt()),
            "rank": int(s.numel()),
            "singular_values": [float(x) for x in s.tolist()],
        }
    return stats


# ------------------------------------------------------------------------------- jobs
def make_run_id(cell, seed, cfg, layer, rank, epochs, lr, method="lora", alpha_mode="2r", data_fraction=1.0):
    """Backward compatible with v2 run_id: only non-default method / alpha_mode / fraction are appended."""
    base = f"{cell}__s{seed}__{cfg.replace(':', '-')}__L{layer}__r{rank}__e{epochs}__lr{lr:g}"
    if method != "lora":       base += f"__m{method}"
    if alpha_mode != "2r":     base += f"__a{alpha_mode.replace(':', '')}"
    if data_fraction < 1.0:    base += f"__df{data_fraction:g}"
    return base


def ranks_for(cfg, layer, rank, base_rank, L=12):
    layer_ranks, type_ranks = [base_rank] * L, None
    if cfg == "layer":
        layer_ranks[layer] = rank
    elif cfg.startswith("group:"):
        for i in GROUPS[cfg.split(":", 1)[1]]:
            layer_ranks[i] = rank
    elif cfg == "global":
        layer_ranks = [rank] * L
    elif cfg.startswith("type:"):
        type_ranks = {cfg.split(":", 1)[1]: rank}
    return layer_ranks, type_ranks


def build_jobs(args):
    seeds = args.seeds or [args.seed]
    epochs_list = args.epochs_list or [args.epochs]
    lr_list = args.lr_list or [args.lr]
    sweeps = args.sweep or args.sweep_groups or args.sweep_types or args.sweep_global
    jobs = []
    for s in seeds:
        for ep in epochs_list:
            for lr in lr_list:
                def add(cfg, layer, rank):
                    jobs.append(dict(seed=s, cfg=cfg, layer=layer, rank=rank, epochs=ep, lr=lr))
                if args.baseline or (not sweeps and args.config == "baseline"):
                    add("baseline", -1, args.base_rank)
                if args.sweep:
                    for m in args.layers:
                        for r in args.ranks: add("layer", m, r)
                if args.sweep_groups:
                    for g in args.groups:
                        for r in args.ranks: add(f"group:{g}", -1, r)
                if args.sweep_types:
                    for t in (args.module_types or MODULE_TYPES):
                        for r in args.ranks: add(f"type:{t}", -1, r)
                if args.sweep_global:
                    for r in args.ranks: add("global", -1, r)
                if not sweeps:
                    if args.config == "layer": add("layer", args.layer, args.rank)
                    elif args.config == "group": add(f"group:{args.group}", -1, args.rank)
                    elif args.config == "type": add(f"type:{args.module_type}", -1, args.rank)
                    elif args.config == "global": add("global", -1, args.rank)
    for j in jobs:
        j["run_id"] = make_run_id(args.cell, j["seed"], j["cfg"], j["layer"], j["rank"], j["epochs"], j["lr"],
                                  method=args.method, alpha_mode=args.alpha_mode,
                                  data_fraction=args.data_fraction)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        assert 0 <= i < n, "--shard must be i/n with 0 <= i < n"
        seeds_all = list(dict.fromkeys(j["seed"] for j in jobs))
        if len(seeds_all) >= n:
            mine = {sd for k, sd in enumerate(seeds_all) if k % n == i}
            jobs = [j for j in jobs if j["seed"] in mine]
        else:
            jobs = [j for k, j in enumerate(jobs) if k % n == i]
    ids = [j["run_id"] for j in jobs]
    assert len(ids) == len(set(ids)), "duplicate run_id in job list"
    return jobs


def read_done(path):
    if not os.path.exists(path): return set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames and "run_id" not in rd.fieldnames:
            raise SystemExit(f"{path} has no run_id column (v1 file). Use a NEW --out file for v3 runs.")
        return {r["run_id"] for r in rd}


# -------------------------------------------------------------------------------- model
def build_model(args, K, job):
    """Return (model, n_trainable_params, peft_model_or_None)."""
    from transformers import AutoModelForSequenceClassification
    set_seed(job["seed"])
    base = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=K)

    if args.method == "full":
        n = sum(p.numel() for p in base.parameters() if p.requires_grad)
        return base, n, None

    if args.method == "head":
        for p in base.parameters():
            p.requires_grad = False
        trainable = 0
        for name, p in base.named_parameters():
            if name.startswith("classifier.") or name.startswith("score."):
                p.requires_grad = True
                trainable += p.numel()
        if trainable == 0:
            raise SystemExit("--method head: could not locate `classifier.` or `score.` params")
        return base, trainable, None

    # lora (default)
    from peft import get_peft_model
    layer_ranks, type_ranks = ranks_for(job["cfg"], job["layer"], job["rank"], args.base_rank)
    cfg = rankcap.lora_config(base, layer_ranks, base_r=args.base_rank, type_ranks=type_ranks,
                              alpha_mode=args.alpha_mode)
    peft_model = get_peft_model(base, cfg)
    return peft_model, rankcap.count_lora_params(peft_model), peft_model


# -------------------------------------------------------------------------------- run
def run_one(args, tok, K, loaders, job, class_weights, precision):
    model, n_params, peft_model = build_model(args, K, job)
    train_dl = make_loader(tok, loaders["train"], args.batch_size, True, job["seed"])
    cal_dl = make_loader(tok, loaders["cal"], 64, False, 0) if loaders["cal"] is not None else None
    eval_dl = make_loader(tok, loaders["eval"], 64, False, 0)
    _, amp_dtype = precision
    t0 = time.time()
    final, epochwise = train_and_eval(model, train_dl, cal_dl, eval_dl, job["epochs"], job["lr"], args.device,
                                      amp_dtype, K, class_weights, args.epoch_eval)
    sec = time.time() - t0
    dw = delta_w_stats(peft_model) if peft_model is not None else {}
    del model, peft_model
    if args.device == "cuda": torch.cuda.empty_cache()
    return final, epochwise, n_params, sec, dw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--cell", default="C1")
    ap.add_argument("--base_rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--epochs_list", type=int, nargs="*")
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--lr_list", type=float, nargs="*")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--eval_file", default="test.csv")
    ap.add_argument("--cal_file", default="calibration.csv")
    ap.add_argument("--epoch_eval", choices=["none", "cal", "both"], default="cal")
    ap.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    ap.add_argument("--class_weight", choices=["none", "balanced"], default="none")
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=100); ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--config", choices=["baseline", "layer", "type", "group", "global"], default="baseline")
    ap.add_argument("--layer", type=int); ap.add_argument("--group", choices=list(GROUPS))
    ap.add_argument("--module_type"); ap.add_argument("--rank", type=int)
    ap.add_argument("--sweep", action="store_true"); ap.add_argument("--sweep_groups", action="store_true")
    ap.add_argument("--sweep_types", action="store_true"); ap.add_argument("--sweep_global", action="store_true")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--ranks", type=int, nargs="*", default=[1, 4, 8, 16, 32])  # v3: rank 8 by default
    ap.add_argument("--layers", type=int, nargs="*", default=list(range(12)))
    ap.add_argument("--groups", nargs="*", default=list(GROUPS))
    ap.add_argument("--module_types", nargs="*")
    ap.add_argument("--shard", default=None)
    # --- v3
    ap.add_argument("--method", choices=["lora", "full", "head"], default="lora")
    ap.add_argument("--alpha_mode", default="2r",
                    help="'2r' (default), 'fixed:N' (e.g. fixed:16), or 'rslora'")
    ap.add_argument("--data_fraction", type=float, default=1.0,
                    help="stratified subsample of training set, in (0,1]; the value enters run_id")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    assert 0 < args.data_fraction <= 1.0, "--data_fraction must be in (0,1]"

    jobs = build_jobs(args)
    done = read_done(args.out)
    todo = [j for j in jobs if j["run_id"] not in done]
    print(f"{len(jobs)} jobs in this shard, {len(jobs) - len(todo)} already done, {len(todo)} to run", flush=True)
    if args.dry_run:
        for j in todo: print(j["run_id"])
        return
    if not todo: return

    from transformers import AutoTokenizer
    import transformers
    try:
        import peft; peft_v = peft.__version__
    except Exception:
        peft_v = "?"
    tok = AutoTokenizer.from_pretrained(args.model)

    # --- training data (with optional stratified fraction)
    tr_texts, tr_labels = load_csv(os.path.join(args.data_dir, "train.csv"))
    if args.limit_train:
        tr_texts, tr_labels = tr_texts[:args.limit_train], tr_labels[:args.limit_train]
    # data_fraction uses the FIRST job's seed to define the subsample (kept identical across the whole invocation)
    seed_for_fraction = jobs[0]["seed"] if jobs else args.seed
    if args.data_fraction < 1.0:
        n_before = len(tr_labels)
        tr_texts, tr_labels = stratified_subsample(tr_texts, tr_labels, args.data_fraction, seed_for_fraction)
        print(f"data_fraction={args.data_fraction}: kept {len(tr_labels)} / {n_before} train rows "
              f"(seed {seed_for_fraction} for stratification)", flush=True)

    ev = load_csv(os.path.join(args.data_dir, args.eval_file))
    cal_path = os.path.join(args.data_dir, args.cal_file)
    cal = load_csv(cal_path) if os.path.exists(cal_path) else None
    if cal is None: print(f"WARNING: {cal_path} not found, calibration metrics will be empty", flush=True)
    K = len(set(tr_labels))
    loaders = {"train": tokenize_split(tok, tr_texts, tr_labels, args.max_len),
               "eval": tokenize_split(tok, *ev, args.max_len),
               "cal": tokenize_split(tok, *cal, args.max_len) if cal is not None else None}
    class_weights = None
    if args.class_weight == "balanced":
        cnt = np.bincount(tr_labels, minlength=K).astype(float)
        class_weights = torch.tensor(len(tr_labels) / (K * cnt), dtype=torch.float32)
    precision = resolve_precision(args.device, args.precision)
    gh = git_hash()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    for sub in ("logits", "epochwise", "deltaw"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    with open(args.out + ".meta.jsonl", "a", encoding="utf-8") as mf:
        mf.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "git": gh, "torch": torch.__version__,
                             "transformers": transformers.__version__, "peft": peft_v, "model": args.model,
                             "device": args.device,
                             "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
                             "compute_capability": list(torch.cuda.get_device_capability(0)) if args.device == "cuda" else None,
                             "precision": precision[0], "class_weight": args.class_weight,
                             "method": args.method, "alpha_mode": args.alpha_mode,
                             "data_fraction": args.data_fraction, "n_train": len(tr_labels),
                             "argv": sys.argv[1:]}) + "\n")

    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new: w.writeheader()
        for j in todo:
            final, epochwise, n_params, sec, dw = run_one(args, tok, K, loaders, j, class_weights, precision)
            m, lg_e, gd_e = final["eval"]
            row = dict(run_id=j["run_id"], cell=args.cell, seed=j["seed"], config=j["cfg"], layer=j["layer"],
                       rank=j["rank"], epochs=j["epochs"], lr=j["lr"], batch_size=args.batch_size,
                       max_len=args.max_len, precision=precision[0], class_weight=args.class_weight,
                       limit_train=args.limit_train, method=args.method, alpha_mode=args.alpha_mode,
                       data_fraction=args.data_fraction, n_train=len(tr_labels),
                       macro_f1=f"{m['macro_f1']:.6f}", accuracy=f"{m['accuracy']:.6f}", nll=f"{m['nll']:.6f}",
                       f1_per_class=";".join(f"{x:.6f}" for x in m["per_class"]),
                       lora_params=n_params, seconds=f"{sec:.1f}", eval_file=args.eval_file, git=gh)
            if K == 3:
                row.update(f1_neg=f"{m['per_class'][0]:.6f}", f1_neu=f"{m['per_class'][1]:.6f}",
                           f1_pos=f"{m['per_class'][2]:.6f}")
            arrays = {"eval_logits": lg_e, "eval_gold": gd_e}
            if "cal" in final:
                mc, lg_c, gd_c = final["cal"]
                row.update(cal_macro_f1=f"{mc['macro_f1']:.6f}", cal_accuracy=f"{mc['accuracy']:.6f}",
                           cal_nll=f"{mc['nll']:.6f}")
                arrays.update(cal_logits=lg_c, cal_gold=gd_c)
            np.savez_compressed(os.path.join(out_dir, "logits", j["run_id"] + ".npz"), **arrays)
            with open(os.path.join(out_dir, "epochwise", j["run_id"] + ".json"), "w", encoding="utf-8") as jf:
                json.dump({"run_id": j["run_id"], "epochs": epochwise}, jf)
            if dw:
                with open(os.path.join(out_dir, "deltaw", j["run_id"] + ".json"), "w", encoding="utf-8") as df:
                    json.dump({"run_id": j["run_id"], "layers": dw}, df)
            w.writerow(row); f.flush()
            print(f"{j['run_id']} f1={m['macro_f1']:.4f} nll={m['nll']:.4f} params={n_params} {sec:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
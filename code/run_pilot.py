"""run_pilot.py v2 - train one LoRA configuration per job and append its metrics to a CSV.

What is new in v2 (everything else behaves like v1)
  * --sweep_global   : the SAME rank r on all 12 layers x 6 module types (global rank sweep).
  * --epochs_list / --lr_list : several training budgets / learning rates in one invocation.
  * --shard i/n      : deterministic split BY SEED (complete designs per GPU) so two GPUs can run two processes
                       (CUDA_VISIBLE_DEVICES=0 ... --shard 0/2 --out a.csv ; =1 ... --shard 1/2 --out b.csv).
  * Every run is evaluated on BOTH calibration.csv and the primary eval file (test.csv by default);
    selection of anything (LR, epoch, temperature) must use calibration only.
  * Per-epoch calibration metrics -> epochwise/<run_id>.json ; final logits + gold labels of both
    splits -> logits/<run_id>.npz (needed for temperature scaling / ECE / Paper B).
  * run_id (unique per seed/config/rank/epochs/lr) is the resume key, so runs with different
    budgets can never be skipped or overwritten by mistake.
  * Reproducibility metadata: precision, class_weight, git hash, library versions (<out>.meta.jsonl).
  * --precision {auto,bf16,fp16,fp32}; auto == v1 behaviour (bf16 if torch says it is supported).
  * --class_weight {none,balanced}; none == what v1 really did (plain cross-entropy).
  * --dry_run prints the job list without training.

Examples
  python run_pilot.py --data_dir data/vsfc_sent --out results/global_gpu0.csv --cell G1 \
      --sweep_global --ranks 1 4 16 32 --epochs_list 3 6 --seeds 200 201 202 203 204 205 206 207 --shard 0/2
  python run_pilot.py ... --config type --module_type intermediate.dense --rank 32 --seed 105
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
    "precision", "class_weight", "limit_train",
    "macro_f1", "f1_neg", "f1_neu", "f1_pos", "f1_per_class", "accuracy", "nll",      # primary eval file
    "cal_macro_f1", "cal_accuracy", "cal_nll",                                          # calibration.csv
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


def tokenize_split(tok, texts, labels, max_len):
    enc = tok(texts, truncation=True, max_length=max_len, padding=False)
    return [{"input_ids": ids, "attention_mask": am, "labels": y}
            for ids, am, y in zip(enc["input_ids"], enc["attention_mask"], labels)]


def make_loader(tok, items, bs, shuffle, seed):
    """Dynamic padding: each batch is padded to its own longest sequence."""
    def collate(batch):
        ys = torch.tensor([b["labels"] for b in batch])
        pad = tok.pad([{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch],
                      padding=True, return_tensors="pt")
        return pad["input_ids"], pad["attention_mask"], ys

    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(items, batch_size=bs, shuffle=shuffle, generator=g if shuffle else None,
                      collate_fn=collate)


def resolve_precision(device, choice):
    """Return (name, autocast dtype or None). 'auto' reproduces v1: bf16 whenever torch reports support."""
    if choice == "auto":
        choice = "bf16" if (device == "cuda" and torch.cuda.is_bf16_supported()) else "fp32"
    if choice == "bf16":
        if device == "cuda" and torch.cuda.get_device_capability(0)[0] < 8:
            print("WARNING: bf16 requested on a GPU without native bf16 (e.g. T4): it is emulated and slow. "
                  "Consider --precision fp16, but never mix precisions inside one comparison.", flush=True)
        return "bf16", torch.bfloat16
    if choice == "fp16":
        if device != "cuda":
            raise SystemExit("--precision fp16 needs a CUDA device")
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
        rec = {"epoch": ep, "train_loss": run_loss / max(1, nb)}
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


# ------------------------------------------------------------------------------- jobs
def make_run_id(cell, seed, cfg, layer, rank, epochs, lr):
    return f"{cell}__s{seed}__{cfg.replace(':', '-')}__L{layer}__r{rank}__e{epochs}__lr{lr:g}"


def ranks_for(cfg, layer, rank, base_rank, L=12):
    """Return (layer_ranks, type_ranks) for a config name."""
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
    """Ordered list of job dicts. Order: seed -> epochs -> lr -> configs, so partial results are balanced."""
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
        j["run_id"] = make_run_id(args.cell, j["seed"], j["cfg"], j["layer"], j["rank"], j["epochs"], j["lr"])
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        assert 0 <= i < n, "--shard must be i/n with 0 <= i < n"
        seeds_all = list(dict.fromkeys(j["seed"] for j in jobs))
        if len(seeds_all) >= n:   # preferred: shard by SEED so every shard holds complete designs
            mine = {sd for k, sd in enumerate(seeds_all) if k % n == i}
            jobs = [j for j in jobs if j["seed"] in mine]
        else:                     # fewer seeds than shards: fall back to job index
            jobs = [j for k, j in enumerate(jobs) if k % n == i]
    ids = [j["run_id"] for j in jobs]
    assert len(ids) == len(set(ids)), "duplicate run_id in job list"
    return jobs


def read_done(path):
    if not os.path.exists(path): return set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames and "run_id" not in rd.fieldnames:
            raise SystemExit(f"{path} has no run_id column (v1 file). Use a NEW --out file for v2 runs.")
        return {r["run_id"] for r in rd}


# -------------------------------------------------------------------------------- run
def run_one(args, tok, K, loaders, job, class_weights, precision):
    from peft import get_peft_model
    from transformers import AutoModelForSequenceClassification
    set_seed(job["seed"])  # same seed => same classifier-head init and LoRA init for every config
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=K)
    layer_ranks, type_ranks = ranks_for(job["cfg"], job["layer"], job["rank"], args.base_rank)
    model = get_peft_model(model, rankcap.lora_config(model, layer_ranks, base_r=args.base_rank,
                                                      type_ranks=type_ranks))
    n_params = rankcap.count_lora_params(model)
    train_dl = make_loader(tok, loaders["train"], args.batch_size, True, job["seed"])
    cal_dl = make_loader(tok, loaders["cal"], 64, False, 0) if loaders["cal"] is not None else None
    eval_dl = make_loader(tok, loaders["eval"], 64, False, 0)
    _, amp_dtype = precision
    t0 = time.time()
    final, epochwise = train_and_eval(model, train_dl, cal_dl, eval_dl, job["epochs"], job["lr"], args.device,
                                      amp_dtype, K, class_weights, args.epoch_eval)
    sec = time.time() - t0
    del model
    if args.device == "cuda": torch.cuda.empty_cache()
    return final, epochwise, n_params, sec


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
    ap.add_argument("--eval_file", default="test.csv", help="primary evaluation file in data_dir")
    ap.add_argument("--cal_file", default="calibration.csv", help="calibration file (skipped if missing)")
    ap.add_argument("--epoch_eval", choices=["none", "cal", "both"], default="cal")
    ap.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    ap.add_argument("--class_weight", choices=["none", "balanced"], default="none")
    ap.add_argument("--limit_train", type=int, default=0, help="debug only: use the first N training rows")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=100); ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--config", choices=["baseline", "layer", "type", "group", "global"], default="baseline")
    ap.add_argument("--layer", type=int); ap.add_argument("--group", choices=list(GROUPS))
    ap.add_argument("--module_type"); ap.add_argument("--rank", type=int)
    ap.add_argument("--sweep", action="store_true", help="per-layer sweep")
    ap.add_argument("--sweep_groups", action="store_true"); ap.add_argument("--sweep_types", action="store_true")
    ap.add_argument("--sweep_global", action="store_true")
    ap.add_argument("--baseline", action="store_true", help="also run uniform r=base_rank per (seed, epochs, lr)")
    ap.add_argument("--ranks", type=int, nargs="*", default=[1, 32])
    ap.add_argument("--layers", type=int, nargs="*", default=list(range(12)))
    ap.add_argument("--groups", nargs="*", default=list(GROUPS))
    ap.add_argument("--module_types", nargs="*")
    ap.add_argument("--shard", default=None, help="i/n : split by seed (k-th seed goes to shard k %% n); by job index if fewer seeds than shards")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

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
    tr = load_csv(os.path.join(args.data_dir, "train.csv"))
    if args.limit_train: tr = (tr[0][:args.limit_train], tr[1][:args.limit_train])
    ev = load_csv(os.path.join(args.data_dir, args.eval_file))
    cal_path = os.path.join(args.data_dir, args.cal_file)
    cal = load_csv(cal_path) if os.path.exists(cal_path) else None
    if cal is None: print(f"WARNING: {cal_path} not found, calibration metrics will be empty", flush=True)
    K = len(set(tr[1]))
    loaders = {"train": tokenize_split(tok, *tr, args.max_len), "eval": tokenize_split(tok, *ev, args.max_len),
               "cal": tokenize_split(tok, *cal, args.max_len) if cal is not None else None}
    class_weights = None
    if args.class_weight == "balanced":
        cnt = np.bincount(tr[1], minlength=K).astype(float)
        class_weights = torch.tensor(len(tr[1]) / (K * cnt), dtype=torch.float32)
    precision = resolve_precision(args.device, args.precision)
    gh = git_hash()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    for sub in ("logits", "epochwise"): os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    with open(args.out + ".meta.jsonl", "a", encoding="utf-8") as mf:
        mf.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "git": gh, "torch": torch.__version__,
                             "transformers": transformers.__version__, "peft": peft_v, "model": args.model,
                             "device": args.device,
                             "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
                             "precision": precision[0], "class_weight": args.class_weight,
                             "argv": sys.argv[1:]}) + "\n")

    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new: w.writeheader()
        for j in todo:
            final, epochwise, n_params, sec = run_one(args, tok, K, loaders, j, class_weights, precision)
            m, lg_e, gd_e = final["eval"]
            row = dict(run_id=j["run_id"], cell=args.cell, seed=j["seed"], config=j["cfg"], layer=j["layer"],
                       rank=j["rank"], epochs=j["epochs"], lr=j["lr"], batch_size=args.batch_size,
                       max_len=args.max_len, precision=precision[0], class_weight=args.class_weight,
                       limit_train=args.limit_train, macro_f1=f"{m['macro_f1']:.6f}",
                       accuracy=f"{m['accuracy']:.6f}", nll=f"{m['nll']:.6f}",
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
            w.writerow(row); f.flush()
            print(f"{j['run_id']} f1={m['macro_f1']:.4f} nll={m['nll']:.4f} params={n_params} {sec:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()

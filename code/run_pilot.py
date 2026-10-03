import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""Train one configuration and append its test Macro-F1 to a CSV.

Examples
  # baseline (uniform r=8) for seed 100
  python run_pilot.py --data_dir data/vsfc_sent --model FacebookAI/xlm-roberta-base --seed 100 --config baseline
  # layer 3 at rank 1, other layers at 8
  python run_pilot.py ... --config layer --layer 3 --rank 1
  # full pilot (all layers x ranks {1,32} x seeds) -> see --sweep
  python run_pilot.py ... --sweep --seeds 100 101 102 103 104 --ranks 1 32

data_dir must hold train.csv and test.csv with columns: text,label (label = int 0..K-1).
Everything that must be paired across configurations (split, init of the classifier head,
data order) is derived from --seed only.
"""
import argparse, csv, os, random, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

import rankcap


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:   # utf-8-sig: tolerate a BOM
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [int(r["label"]) for r in rows]


def make_loader(tok, texts, labels, max_len, bs, shuffle, seed):
    """Dynamic padding: each batch is padded to its own longest sequence (not to the longest in the set)."""
    enc = tok(texts, truncation=True, max_length=max_len, padding=False)
    items = [{"input_ids": ids, "attention_mask": am, "labels": y}
             for ids, am, y in zip(enc["input_ids"], enc["attention_mask"], labels)]

    def collate(batch):
        ys = torch.tensor([b["labels"] for b in batch])
        pad = tok.pad([{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch],
                      padding=True, return_tensors="pt")
        return pad["input_ids"], pad["attention_mask"], ys

    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(items, batch_size=bs, shuffle=shuffle, generator=g if shuffle else None, collate_fn=collate)


def train_eval(model, train_dl, test_dl, epochs, lr, device, warmup=0.1, wd=0.01, clip=1.0):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8)
    total = epochs * len(train_dl); warm = max(1, int(warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    model.to(device)
    for ep in range(epochs):
        model.train()
        for ids, mask, y in train_dl:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    model.eval(); preds, gold = [], []
    with torch.no_grad():
        for ids, mask, y in test_dl:
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            preds += logits.argmax(-1).cpu().tolist(); gold += y.tolist()
    return f1_score(gold, preds, average="macro")


def run_one(args, tok, model_name, num_labels, data, seed, layer_ranks, type_ranks, tag):
    from peft import get_peft_model
    from transformers import AutoModelForSequenceClassification
    set_seed(seed)  # same seed => same classifier-head init for every configuration
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    cfg = rankcap.lora_config(model, layer_ranks, base_r=args.base_rank, type_ranks=type_ranks)
    model = get_peft_model(model, cfg)
    n_params = rankcap.count_lora_params(model)
    train_dl = make_loader(tok, *data["train"], args.max_len, args.batch_size, True, seed)
    test_dl = make_loader(tok, *data["test"], args.max_len, 64, False, seed)
    t0 = time.time()
    f1 = train_eval(model, train_dl, test_dl, args.epochs, args.lr, args.device)
    return f1, n_params, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--out", default="results.csv"); ap.add_argument("--cell", default="C1")
    ap.add_argument("--base_rank", type=int, default=8); ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--eval_file", default="test.csv", help="file in data_dir to evaluate on (use calibration.csv for LR/epoch selection)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=100); ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--config", choices=["baseline", "layer", "type"], default="baseline")
    ap.add_argument("--layer", type=int); ap.add_argument("--module_type"); ap.add_argument("--rank", type=int)
    ap.add_argument("--sweep", action="store_true"); ap.add_argument("--ranks", type=int, nargs="*", default=[1, 32])
    ap.add_argument("--layers", type=int, nargs="*", default=list(range(12)))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    tr = load_csv(os.path.join(args.data_dir, "train.csv")); te = load_csv(os.path.join(args.data_dir, args.eval_file))
    data = {"train": tr, "test": te}; K = len(set(tr[1]))
    L = 12

    jobs = []
    seeds = args.seeds or [args.seed]
    for s in seeds:
        if args.sweep:
            jobs.append((s, "baseline", -1, 8))
            jobs += [(s, "layer", m, r) for m in args.layers for r in args.ranks]
        elif args.config == "baseline":
            jobs.append((s, "baseline", -1, 8))
        elif args.config == "layer":
            jobs.append((s, "layer", args.layer, args.rank))
        else:
            jobs.append((s, f"type:{args.module_type}", -1, args.rank))

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for r in csv.DictReader(f):
                done.add((int(r["seed"]), r["config"], int(r["layer"]), int(r["rank"])))
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["cell", "seed", "config", "layer", "rank", "macro_f1", "lora_params", "seconds"])
        for (s, cfg, m, r) in jobs:
            if (s, cfg, m, r) in done:
                continue  # resume support
            layer_ranks = [args.base_rank] * L; type_ranks = None
            if cfg == "layer":
                layer_ranks[m] = r
            elif cfg.startswith("type:"):
                type_ranks = {cfg.split(":", 1)[1]: r}
            f1, n_params, sec = run_one(args, tok, args.model, K, data, s, layer_ranks, type_ranks, cfg)
            w.writerow([args.cell, s, cfg, m, r, f"{f1:.6f}", n_params, f"{sec:.1f}"]); f.flush()
            print(f"seed={s} {cfg} layer={m} rank={r} f1={f1:.4f} params={n_params} {sec:.0f}s", flush=True)


if __name__ == "__main__":
    main()

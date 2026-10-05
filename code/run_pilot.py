"""Train one configuration and append its metrics to a CSV.

Examples
  python run_pilot.py --data_dir data/vsfc_sent --model FacebookAI/xlm-roberta-base --seed 100 --config baseline
  python run_pilot.py ... --config layer --layer 3 --rank 1
  python run_pilot.py ... --config group --group bottom --rank 32
  python run_pilot.py ... --config type --module_type attention.self.query --rank 32
  python run_pilot.py ... --sweep --seeds 100 101 102 103 104 --ranks 1 32
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse, csv, os, random, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, log_loss

import rankcap


# Layer-group definitions
GROUPS = {
    'bottom': [0, 1, 2, 3],
    'middle': [4, 5, 6, 7],
    'top':    [8, 9, 10, 11],
}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [int(r["label"]) for r in rows]


def make_loader(tok, texts, labels, max_len, bs, shuffle, seed):
    """Dynamic padding: each batch padded to its own longest sequence."""
    enc = tok(texts, truncation=True, max_length=max_len, padding=False)
    items = [{"input_ids": ids, "attention_mask": am, "labels": y}
             for ids, am, y in zip(enc["input_ids"], enc["attention_mask"], labels)]

    def collate(batch):
        ys = torch.tensor([b["labels"] for b in batch])
        pad = tok.pad(
            [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
             for b in batch],
            padding=True, return_tensors="pt"
        )
        return pad["input_ids"], pad["attention_mask"], ys

    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(items, batch_size=bs, shuffle=shuffle,
                      generator=g if shuffle else None, collate_fn=collate)


def train_eval(model, train_dl, test_dl, epochs, lr, device,
               warmup=0.1, wd=0.01, clip=1.0, num_classes=3):
    """Train and evaluate. Return dict of metrics + logits array."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd,
                            betas=(0.9, 0.999), eps=1e-8)
    total = epochs * len(train_dl); warm = max(1, int(warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm)))
    )
    use_bf16 = (device == "cuda" and torch.cuda.is_bf16_supported())
    model.to(device)

    for ep in range(epochs):
        model.train()
        for ids, mask, y in train_dl:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=use_bf16):
                loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

    model.eval()
    preds, gold, all_logits = [], [], []
    with torch.no_grad():
        for ids, mask, y in test_dl:
            logits = model(input_ids=ids.to(device),
                           attention_mask=mask.to(device)).logits
            all_logits.append(logits.float().cpu())
            preds += logits.argmax(-1).cpu().tolist()
            gold += y.tolist()

    logits_arr = torch.cat(all_logits, dim=0).numpy()
    labels_list = list(range(num_classes))

    f1_macro = f1_score(gold, preds, average="macro", labels=labels_list,
                        zero_division=0)
    per_class = f1_score(gold, preds, average=None, labels=labels_list,
                         zero_division=0).tolist()
    acc = accuracy_score(gold, preds)

    # NLL: softmax logits, log_loss
    probs = torch.softmax(torch.from_numpy(logits_arr), dim=-1).numpy()
    try:
        nll = log_loss(gold, probs, labels=labels_list)
    except Exception:
        nll = float("nan")

    return {
        "macro_f1": float(f1_macro),
        "f1_neg":  float(per_class[0]),
        "f1_neu":  float(per_class[1]),
        "f1_pos":  float(per_class[2]),
        "accuracy": float(acc),
        "nll": float(nll),
        "logits": logits_arr,
        "gold": gold,
    }


def run_one(args, tok, model_name, num_labels, data, seed,
            layer_ranks, type_ranks, tag):
    from peft import get_peft_model
    from transformers import AutoModelForSequenceClassification
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels)
    cfg = rankcap.lora_config(model, layer_ranks,
                              base_r=args.base_rank, type_ranks=type_ranks)
    model = get_peft_model(model, cfg)
    n_params = rankcap.count_lora_params(model)

    train_dl = make_loader(tok, *data["train"], args.max_len,
                           args.batch_size, True, seed)
    test_dl = make_loader(tok, *data["test"], args.max_len,
                          64, False, seed)
    t0 = time.time()
    metrics = train_eval(model, train_dl, test_dl,
                         args.epochs, args.lr, args.device,
                         num_classes=num_labels)
    elapsed = time.time() - t0

    # Save logits
    logits_dir = os.path.join(os.path.dirname(args.out) or ".", "logits")
    os.makedirs(logits_dir, exist_ok=True)
    safe_tag = tag.replace(":", "-").replace(".", "_")
    logits_path = os.path.join(
        logits_dir,
        f"{args.cell}_{seed}_{safe_tag}_L{m_placeholder(layer_ranks) if False else ''}"
        f"{'_' + str(args.layer) if args.config == 'layer' else ''}"
        f"{'_' + args.group if args.config == 'group' else ''}"
        f"_{args.rank if args.rank is not None else 'base'}.npy"
    )
    np.save(logits_path, metrics["logits"])

    return metrics, n_params, elapsed


def m_placeholder(*a, **k):
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--cell", default="C1")
    ap.add_argument("--base_rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--eval_file", default="test.csv",
                    help="file in data_dir to evaluate on")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--config",
                    choices=["baseline", "layer", "type", "group"],
                    default="baseline")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--group", choices=list(GROUPS.keys()))
    ap.add_argument("--module_type")
    ap.add_argument("--rank", type=int)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--ranks", type=int, nargs="*", default=[1, 32])
    ap.add_argument("--layers", type=int, nargs="*", default=list(range(12)))
    ap.add_argument("--groups", nargs="*", default=list(GROUPS.keys()))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    tr = load_csv(os.path.join(args.data_dir, "train.csv"))
    te = load_csv(os.path.join(args.data_dir, args.eval_file))
    data = {"train": tr, "test": te}
    K = len(set(tr[1]))
    L = 12

    jobs = []
    seeds = args.seeds or [args.seed]
    for s in seeds:
        if args.sweep:
            jobs.append((s, "baseline", -1, 8))
            jobs += [(s, "layer", m, r)
                     for m in args.layers for r in args.ranks]
        elif args.config == "baseline":
            jobs.append((s, "baseline", -1, 8))
        elif args.config == "layer":
            jobs.append((s, "layer", args.layer, args.rank))
        elif args.config == "group":
            jobs.append((s, f"group:{args.group}", -1, args.rank))
        else:  # type
            jobs.append((s, f"type:{args.module_type}", -1, args.rank))

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                done.add((int(r["seed"]), r["config"],
                          int(r["layer"]), int(r["rank"])))

    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow([
                "cell", "seed", "config", "layer", "rank",
                "macro_f1", "f1_neg", "f1_neu", "f1_pos",
                "accuracy", "nll",
                "lora_params", "seconds", "eval_file"
            ])
        for (s, cfg, m, r) in jobs:
            if (s, cfg, m, r) in done:
                continue
            layer_ranks = [args.base_rank] * L
            type_ranks = None
            if cfg == "layer":
                layer_ranks[m] = r
            elif cfg.startswith("group:"):
                g = cfg.split(":", 1)[1]
                for mm in GROUPS[g]:
                    layer_ranks[mm] = r
            elif cfg.startswith("type:"):
                type_ranks = {cfg.split(":", 1)[1]: r}

            metrics, n_params, sec = run_one(
                args, tok, args.model, K, data, s, layer_ranks,
                type_ranks, cfg)

            w.writerow([
                args.cell, s, cfg, m, r,
                f"{metrics['macro_f1']:.6f}",
                f"{metrics['f1_neg']:.6f}",
                f"{metrics['f1_neu']:.6f}",
                f"{metrics['f1_pos']:.6f}",
                f"{metrics['accuracy']:.6f}",
                f"{metrics['nll']:.6f}",
                n_params, f"{sec:.1f}", args.eval_file
            ])
            f.flush()
            print(f"seed={s} {cfg} layer={m} rank={r} "
                  f"f1={metrics['macro_f1']:.4f} nll={metrics['nll']:.4f} "
                  f"params={n_params} {sec:.0f}s", flush=True)


if __name__ == "__main__":
    main()
"""run_adalora.py v2 - AdaLoRA baseline cho so sanh "Adaptive Rank vs Global Rank".

Chay CUNG giao thuc voi run_pilot.py de so sanh cong bang (xem README, muc "Quyet dinh thiet ke"):
  * du lieu train/calibration/test, max_len 128, dynamic padding, batch 32, ham load/token hoa cua run_pilot
  * 72 module dich duoc liet ke tuong minh (6 module x 12 layer, rankcap.module_names),
    KHONG dung suffix "dense" nen khong cham vao classifier.dense / pooler
  * AdamW (wd 0.01), warmup 10% + cosine, clip grad 1.0, fp16 autocast + GradScaler
  * cross-entropy khong trong so (+ orthogonal regularization cua AdaLoRA do PEFT cong vao loss)
  * set_seed day du (random/numpy/torch/cuda) TRUOC khi tao model -> cung seed thi cung khoi tao classifier
    voi cac luot LoRA, ghep cap theo seed duoc
  * metric/CSV cung cot voi run_pilot (macro_f1, f1 theo lop, accuracy, nll, cal_*), run_id de resume,
    luu logits/epochwise de phan tich hieu chuan sau nay

Dac thu AdaLoRA (da doi chieu voi peft 0.21 trong tuners/adalora):
  * init_rank > target_rank. target_rank la rank TRUNG BINH cuoi cung moi module (target_bgt = target_r * 72).
    Muon ngan sach khop LoRA global rank 8 thi dung target_rank 8, init_rank 12.
  * update_and_allocate() phai goi SAU optimizer.step() va TRUOC zero_grad() (can p.grad).
  * Voi GradScaler fp16, luot bi bo qua do overflow co grad inf/nan -> zero grad truoc khi cap nhat do quan trong
    de khong lam hong EMA.
  * Trong PEFT, `tfinal` la SO BUOC CUOI CUNG gia tri rank bi co dinh (khong phai moc bat dau). Cat tia dien ra
    trong khoang (tinit, total_step - tfinal]. Mac dinh tinit=10%, tfinal=40% tong buoc -> cat tia tu 10% den 60%.
  * lora_alpha = 2 * init_rank de he so scaling alpha/r = 2 nhu cac luot LoRA (alpha = 2r).
  * Loss lay tu outputs.loss (truyen labels vao model) vi PEFT chi cong orthogonal regularization
    (orth_reg_weight, mac dinh 0.5) khi model tu tinh loss.

Vi du (Kaggle, 1 GPU):
  python run_adalora.py --data_dir /kaggle/input/datasets/lequangphat260206/peft-data \\
      --out /kaggle/working/results/global/adalora_results.csv \\
      --seeds 300 301 302 --epochs 6 --lr 2e-4 --init_rank 12 --target_rank 8
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse, csv, json, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import rankcap
import run_pilot as rp

CSV_COLUMNS = rp.CSV_COLUMNS + [
    "init_rank", "target_rank", "tinit", "tfinal", "deltaT", "orth_reg_weight",
    "final_rank_sum", "target_budget", "final_rank_min", "final_rank_max", "n_skipped_steps",
]


# ------------------------------------------------------------------------------ jobs
def make_run_id(cell, seed, init_rank, target_rank, epochs, lr, tinit_frac, tfinal_frac):
    return (f"{cell}__s{seed}__adalora__i{init_rank}t{target_rank}__e{epochs}__lr{lr:g}"
            f"__ti{tinit_frac:g}tf{tfinal_frac:g}")


def schedule_steps(total_steps, tinit_frac, tfinal_frac, deltaT):
    """Return (tinit, tfinal) in steps using PEFT semantics (tfinal = number of final fixed-rank steps)."""
    tinit = int(total_steps * tinit_frac)
    tfinal = int(total_steps * tfinal_frac)
    prune_window = total_steps - tfinal - tinit
    if prune_window < deltaT:
        raise ValueError(
            f"Cua so cat tia = total_step - tfinal - tinit = {prune_window} < deltaT = {deltaT}. "
            "Giam --tinit_frac/--tfinal_frac hoac --deltaT.")
    return tinit, tfinal


def build_jobs(args):
    seeds = args.seeds
    epochs_list = args.epochs_list or [args.epochs]
    lr_list = args.lr_list or [args.lr]
    lr_map = None
    if args.lr_map:
        with open(args.lr_map, encoding="utf-8") as f:
            raw = json.load(f)
        lr_map = {int(k): float(v["best_lr"] if isinstance(v, dict) else v) for k, v in raw.items()}
        if args.target_rank not in lr_map:
            raise SystemExit(f"--lr_map khong co rank {args.target_rank}: {sorted(lr_map)}")
        lr_list = [lr_map[args.target_rank]]
        print(f"[run_adalora] LR lay tu lr_map cho rank {args.target_rank}: {lr_list[0]:g}", flush=True)
    jobs = []
    for s in seeds:
        for ep in epochs_list:
            for lr in lr_list:
                jobs.append(dict(
                    seed=s, epochs=ep, lr=lr,
                    run_id=make_run_id(args.cell, s, args.init_rank, args.target_rank, ep, lr,
                                       args.tinit_frac, args.tfinal_frac)))
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


# ------------------------------------------------------------------------------ model
def build_adalora_model(args, K, seed, total_steps, tinit, tfinal):
    from peft import AdaLoraConfig, TaskType, get_peft_model
    rp.set_seed(seed)
    base = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=K)
    targets = rankcap.module_names(base)  # 6 module/layer, ten day du, khong dung suffix "dense"
    cfg = AdaLoraConfig(
        task_type=TaskType.SEQ_CLS,
        init_r=args.init_rank, target_r=args.target_rank,
        lora_alpha=int(round(args.alpha_ratio * args.init_rank)),
        lora_dropout=args.lora_dropout,
        target_modules=targets, bias="none",
        total_step=total_steps, tinit=tinit, tfinal=tfinal, deltaT=args.deltaT,
        orth_reg_weight=args.orth_reg_weight,
    )
    peft_model = get_peft_model(base, cfg)
    n_adapted = sum(1 for n, _ in peft_model.named_parameters() if ".lora_E." in n)
    assert n_adapted == len(targets), f"AdaLoRA gan vao {n_adapted} module, ky vong {len(targets)}"
    n_lora = sum(p.numel() for n, p in peft_model.named_parameters() if "lora_" in n)
    return peft_model, n_lora, len(targets)


def final_ranks(peft_model):
    """Rank cuoi cung cua tung module = so phan tu lora_E khac 0 (phan tu bi cat bi masked_fill = 0)."""
    out = {}
    for n, p in peft_model.named_parameters():
        if ".lora_E." in n:
            key = n.split(".lora_E.")[0].replace("base_model.model.", "", 1)
            out[key] = int((p.detach().abs() > 0).sum().item())
    return out


# ------------------------------------------------------------------------------ train
def train_and_eval(model, train_dl, cal_dl, eval_dl, epochs, lr, device, amp_dtype, K,
                   epoch_eval="cal", warmup=0.1, wd=0.01, clip=1.0):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8)
    total = epochs * len(train_dl)
    warm = max(1, int(warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    model.to(device)

    epochwise, final = [], {}
    global_step, n_skipped = 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        run_loss, run_ce, nb = 0.0, 0.0, 0
        for ids, mask, y in train_dl:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(input_ids=ids, attention_mask=mask, labels=y)
            loss = out.loss  # CE + orth_reg_weight * orthogonal regularization (do PEFT cong vao)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(params, clip)
            finite = bool(torch.isfinite(gn))
            scaler.step(opt); scaler.update(); sched.step()
            if not finite:
                # GradScaler da bo qua optimizer.step(); grad inf/nan se lam hong EMA do quan trong cua AdaLoRA
                n_skipped += 1
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()
            model.base_model.update_and_allocate(global_step)  # SAU step, TRUOC zero_grad
            opt.zero_grad(set_to_none=True)
            global_step += 1
            run_loss += float(loss.detach())
            run_ce += float(F.cross_entropy(out.logits.detach().float(), y))
            nb += 1
        rec = {"epoch": ep, "train_loss": run_loss / max(1, nb), "train_ce": run_ce / max(1, nb),
               "lr": float(opt.param_groups[0]["lr"])}
        last = ep == epochs
        if cal_dl is not None and (epoch_eval in ("cal", "both") or last):
            m, lg, gd = rp.evaluate(model, cal_dl, device, amp_dtype, K)
            rec.update({"cal_macro_f1": m["macro_f1"], "cal_nll": m["nll"], "cal_accuracy": m["accuracy"]})
            if last: final["cal"] = (m, lg, gd)
        if eval_dl is not None and (epoch_eval == "both" or last):
            m, lg, gd = rp.evaluate(model, eval_dl, device, amp_dtype, K)
            rec.update({"eval_macro_f1": m["macro_f1"], "eval_nll": m["nll"], "eval_accuracy": m["accuracy"]})
            if last: final["eval"] = (m, lg, gd)
        epochwise.append(rec)
        print(f"  epoch {ep}/{epochs} loss={rec['train_loss']:.4f} ce={rec['train_ce']:.4f}"
              + (f" cal_f1={rec['cal_macro_f1']:.4f}" if "cal_macro_f1" in rec else ""), flush=True)
    return final, epochwise, n_skipped


def run_one(args, tok, K, loaders, job, precision):
    train_dl = rp.make_loader(tok, loaders["train"], args.batch_size, True, job["seed"])
    cal_dl = rp.make_loader(tok, loaders["cal"], 64, False, 0) if loaders["cal"] is not None else None
    eval_dl = rp.make_loader(tok, loaders["eval"], 64, False, 0)
    total_steps = job["epochs"] * len(train_dl)
    tinit, tfinal = schedule_steps(total_steps, args.tinit_frac, args.tfinal_frac, args.deltaT)
    print(f"  total_step={total_steps} tinit={tinit} tfinal={tfinal} "
          f"(cat tia trong ({tinit}, {total_steps - tfinal}], deltaT={args.deltaT})", flush=True)
    model, n_lora, n_mod = build_adalora_model(args, K, job["seed"], total_steps, tinit, tfinal)
    t0 = time.time()
    final, epochwise, n_skipped = train_and_eval(
        model, train_dl, cal_dl, eval_dl, job["epochs"], job["lr"], args.device, precision[1], K, args.epoch_eval)
    sec = time.time() - t0
    ranks = final_ranks(model)
    del model
    if args.device == "cuda": torch.cuda.empty_cache()
    return final, epochwise, n_lora, n_mod, sec, ranks, n_skipped, tinit, tfinal


# ------------------------------------------------------------------------------ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="AdaLoRA baseline (cung giao thuc voi run_pilot.py)")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", default="ADA")
    ap.add_argument("--seeds", type=int, nargs="+", default=[300, 301, 302])
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--epochs_list", type=int, nargs="*")
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--lr_list", type=float, nargs="*")
    ap.add_argument("--lr_map", default=None,
                    help="JSON {rank: lr} hoac {rank: {best_lr: lr}} (tuning_decision.json); lay LR cua --target_rank")
    ap.add_argument("--init_rank", type=int, default=12)
    ap.add_argument("--target_rank", type=int, default=8)
    ap.add_argument("--tinit_frac", type=float, default=0.1, help="ty le buoc khoi dong truoc khi cat tia")
    ap.add_argument("--tfinal_frac", type=float, default=0.4,
                    help="ty le buoc CUOI giu rank co dinh (ngu nghia cua PEFT)")
    ap.add_argument("--deltaT", type=int, default=50)
    ap.add_argument("--alpha_ratio", type=float, default=2.0, help="lora_alpha = alpha_ratio * init_rank")
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--orth_reg_weight", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--eval_file", default="test.csv")
    ap.add_argument("--cal_file", default="calibration.csv")
    ap.add_argument("--epoch_eval", choices=["none", "cal", "both"], default="cal")
    ap.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="fp16")
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)
    # AdaLoRA: init_r (rank ban dau, lon) phai > target_r (rank trung binh sau khi cat tia)
    if args.init_rank <= args.target_rank:
        ap.error(f"init_rank ({args.init_rank}) phai lon hon target_rank ({args.target_rank}). "
                 "Vi du: --init_rank 12 --target_rank 8")
    return args


def main(argv=None):
    args = parse_args(argv)
    jobs = build_jobs(args)
    done = rp.read_done(args.out)
    todo = [j for j in jobs if j["run_id"] not in done]
    print(f"{len(jobs)} jobs in this shard, {len(jobs) - len(todo)} already done, {len(todo)} to run", flush=True)
    if args.dry_run:
        for j in todo: print(j["run_id"])
        return
    if not todo:
        return

    import transformers
    try:
        import peft; peft_v = peft.__version__
    except Exception:
        peft_v = "?"
    tok = AutoTokenizer.from_pretrained(args.model)
    tr_texts, tr_labels = rp.load_csv(os.path.join(args.data_dir, "train.csv"))
    if args.limit_train:
        tr_texts, tr_labels = tr_texts[:args.limit_train], tr_labels[:args.limit_train]
    ev = rp.load_csv(os.path.join(args.data_dir, args.eval_file))
    cal_path = os.path.join(args.data_dir, args.cal_file)
    cal = rp.load_csv(cal_path) if os.path.exists(cal_path) else None
    if cal is None: print(f"WARNING: {cal_path} not found, calibration metrics will be empty", flush=True)
    K = len(set(tr_labels))
    loaders = {"train": rp.tokenize_split(tok, tr_texts, tr_labels, args.max_len),
               "eval": rp.tokenize_split(tok, *ev, args.max_len),
               "cal": rp.tokenize_split(tok, *cal, args.max_len) if cal is not None else None}
    precision = rp.resolve_precision(args.device, args.precision)
    gh = rp.git_hash()
    print(f"Device: {args.device} precision={precision[0]} n_train={len(tr_labels)} K={K}", flush=True)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    for sub in ("logits", "epochwise", "adalora_rank"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    with open(args.out + ".meta.jsonl", "a", encoding="utf-8") as mf:
        mf.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "git": gh, "torch": torch.__version__,
                             "transformers": transformers.__version__, "peft": peft_v, "model": args.model,
                             "device": args.device, "precision": precision[0], "method": "adalora",
                             "n_train": len(tr_labels), "argv": sys.argv[1:] if argv is None else list(argv)}) + "\n")

    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new: w.writeheader()
        for j in todo:
            print(f"\n=== AdaLoRA seed {j['seed']} epochs={j['epochs']} lr={j['lr']:g} "
                  f"init={args.init_rank} target={args.target_rank} ===", flush=True)
            final, epochwise, n_lora, n_mod, sec, ranks, n_skipped, tinit, tfinal = run_one(
                args, tok, K, loaders, j, precision)
            m, lg_e, gd_e = final["eval"]
            rvals = list(ranks.values())
            row = dict(run_id=j["run_id"], cell=args.cell, seed=j["seed"], config="adalora", layer=-1,
                       rank=args.target_rank, epochs=j["epochs"], lr=j["lr"], batch_size=args.batch_size,
                       max_len=args.max_len, precision=precision[0], class_weight="none",
                       limit_train=args.limit_train, method="adalora", alpha_mode="2r", data_fraction=1.0,
                       n_train=len(tr_labels),
                       macro_f1=f"{m['macro_f1']:.6f}", accuracy=f"{m['accuracy']:.6f}", nll=f"{m['nll']:.6f}",
                       f1_per_class=";".join(f"{x:.6f}" for x in m["per_class"]),
                       lora_params=n_lora, seconds=f"{sec:.1f}", eval_file=args.eval_file, git=gh,
                       init_rank=args.init_rank, target_rank=args.target_rank, tinit=tinit, tfinal=tfinal,
                       deltaT=args.deltaT, orth_reg_weight=args.orth_reg_weight,
                       final_rank_sum=sum(rvals), target_budget=args.target_rank * n_mod,
                       final_rank_min=min(rvals), final_rank_max=max(rvals), n_skipped_steps=n_skipped)
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
            with open(os.path.join(out_dir, "adalora_rank", j["run_id"] + ".json"), "w", encoding="utf-8") as rf:
                json.dump({"run_id": j["run_id"], "ranks": ranks}, rf)
            w.writerow(row); f.flush()
            print(f"{j['run_id']} f1={m['macro_f1']:.4f} nll={m['nll']:.4f} params={n_lora} "
                  f"rank_sum={sum(rvals)}/{args.target_rank * n_mod} (min {min(rvals)}, max {max(rvals)}) "
                  f"skipped={n_skipped} {sec:.0f}s", flush=True)
    print(f"Da luu ket qua tai: {args.out}")


if __name__ == "__main__":
    main()

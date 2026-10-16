"""
run_phase1a.py - Wrapper chay R1 dot 1 (Giai doan 1a) tren 2 GPU Kaggle.

Chay 2 tien trinh song song (moi GPU mot shard), gop CSV thanh global_merged.csv.

Dung:
    python code/run_phase1a.py ^
        --data_dir /kaggle/input/datasets/lequangphat260206/peft-data ^
        --out_dir  /kaggle/working/results/global

Hoac gioi han seeds:
    python code/run_phase1a.py ... --seeds 200 201 202 203 204 205 206 207

Xem docs/analysis_plan_v2.0.md muc 2.1, 6; docs/roadmap_v2.md muc 4.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def build_cmd(args, shard: str, out_csv: str) -> list:
    return [
        sys.executable, args.runner,
        "--data_dir", args.data_dir,
        "--model", args.model,
        "--cell", args.cell,
        "--sweep_global",
        "--ranks", *[str(r) for r in args.ranks],
        "--epochs_list", *[str(e) for e in args.epochs],
        "--seeds", *[str(s) for s in args.seeds],
        "--lr", args.lr,
        "--precision", args.precision,
        "--eval_file", args.eval_file,
        "--cal_file", args.cal_file,
        "--epoch_eval", args.epoch_eval,
        "--shard", shard,
        "--out", out_csv,
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chay R1 dot 1 (Giai doan 1a) tren 2 GPU Kaggle."
    )
    p.add_argument("--runner", default="/kaggle/working/code/code/run_pilot.py")
    p.add_argument("--workdir", default="/kaggle/working/code/code")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    p.add_argument("--cell", default="R1")
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    p.add_argument("--epochs", type=int, nargs="+", default=[3, 6])
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[200, 201, 202, 203, 204, 205, 206, 207])
    p.add_argument("--lr", default="2e-4",
                   help="Fallback khi chua co --lr_decision")
    p.add_argument("--precision", default="fp16")
    p.add_argument("--eval_file", default="test.csv")
    p.add_argument("--cal_file", default="calibration.csv")
    p.add_argument("--epoch_eval", default="cal")
    p.add_argument("--n_gpu", type=int, default=2)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expected = len(args.seeds) * len(args.ranks) * len(args.epochs)
    print("[phase1a] runner    =", args.runner)
    print("[phase1a] seeds     =", args.seeds)
    print("[phase1a] ranks     =", args.ranks)
    print("[phase1a] epochs    =", args.epochs)
    print("[phase1a] precision =", args.precision)
    print("[phase1a] ky vong   =", expected, "luot tren", args.n_gpu, "GPU")

    procs = []
    t0 = time.time()
    for gpu in range(args.n_gpu):
        shard = "{}/{}".format(gpu, args.n_gpu)
        out_csv = out_dir / "global_gpu{}.csv".format(gpu)
        log_path = out_dir / "global_gpu{}.log".format(gpu)
        log = open(log_path, "w", encoding="utf-8")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        cmd = build_cmd(args, shard, str(out_csv))
        print("[phase1a] GPU{} shard={} -> {}".format(gpu, shard, out_csv))
        print("[phase1a]   cmd:", " ".join(cmd))
        proc = subprocess.Popen(
            cmd, cwd=args.workdir, env=env,
            stdout=log, stderr=subprocess.STDOUT,
        )
        procs.append((gpu, proc, log, out_csv))

    exit_codes = {}
    for gpu, proc, log, out_csv in procs:
        rc = proc.wait()
        log.close()
        elapsed = time.time() - t0
        exit_codes[gpu] = rc
        print("[phase1a] GPU{} exit={} elapsed={:.0f}s".format(gpu, rc, elapsed))

    dfs = []
    for gpu, _, _, out_csv in procs:
        if out_csv.exists():
            df = pd.read_csv(out_csv)
            print("[phase1a] {}: {} rows".format(out_csv.name, len(df)))
            dfs.append(df)
        else:
            print("[phase1a] CANH BAO: {} khong ton tai".format(out_csv))

    if not dfs:
        print("[phase1a] Khong co CSV de gop. Thoat voi ma 1.")
        return 1

    merged = pd.concat(dfs, ignore_index=True)
    if "run_id" in merged.columns:
        before = len(merged)
        merged = merged.drop_duplicates(subset="run_id", keep="last")
        dropped = before - len(merged)
        if dropped:
            print("[phase1a] Bo {} run_id trung (giu ban cuoi)".format(dropped))

    merged_path = out_dir / "global_merged.csv"
    merged.to_csv(merged_path, index=False)
    print("[phase1a] Da luu {}: {} rows (ky vong {})".format(
        merged_path, len(merged), expected))

    if any(rc != 0 for rc in exit_codes.values()):
        print("[phase1a] Mot so tien trinh thoat khac 0:", exit_codes)
    if len(merged) < expected:
        print("[phase1a] CANH BAO: thieu {} luot. Chay lai lenh de resume "
              "(run_id giup bo qua luot da xong).".format(expected - len(merged)))
        return 2

    print("[phase1a] Xong, du luot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
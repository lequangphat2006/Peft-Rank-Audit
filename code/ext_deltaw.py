"""ext_deltaw.py - thống kê ΔW = B @ A: rank hiệu dụng, stable rank, chuẩn Frobenius theo (mô hình, rank).

run_pilot.py đã ghi deltaw/<run_id>.json cho mọi lượt LoRA. Script này gộp các file đó để trả lời bằng
CƠ CHẾ câu hỏi "rank danh nghĩa cao có thực sự được dùng không?":
  - effective_rank / rank danh nghĩa (tỉ lệ sử dụng),
  - stable_rank,
  - fro_norm (độ lớn cập nhật; liên quan trực tiếp đến α = 2r).
Phân tích THĂM DÒ, không thuộc họ xác nhận. Thiếu thư mục deltaw thì bỏ qua và nói rõ.

  python code/ext_deltaw.py --csv_p results/ext/phobert/B1_ext_merged.csv --lr_map_p results/ext/phobert/phobert_lr_map_v2.json \
      --deltaw_p results/ext/phobert/deltaw --csv_x results/global/global_merged.csv \
      --lr_map_x results/tuning/tuning_decision.json --deltaw_x results/global/deltaw --out results/ext/phobert/deltaw_summary.csv
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ext_common import read_csv, load_lr_map, select_branch  # noqa: E402


def one_run(path):
    layers = json.load(open(path, encoding="utf-8"))["layers"]
    eff = [v["effective_rank"] for v in layers.values()]
    st = [v["stable_rank"] for v in layers.values()]
    fr = [v["fro_norm"] for v in layers.values()]
    return dict(eff_rank=float(np.mean(eff)), stable_rank=float(np.mean(st)), fro_norm=float(np.mean(fr)),
                n_modules=len(layers))


def collect(csv, lr_map, dw_dir, tag, epochs=6):
    if not dw_dir or not os.path.isdir(dw_dir):
        print(f"[{tag}] không có thư mục deltaw ({dw_dir}); bỏ qua")
        return None
    df = select_branch(read_csv(csv).query("epochs == @epochs"), load_lr_map(lr_map))
    rows = []
    for _, r in df.iterrows():
        p = os.path.join(dw_dir, r["run_id"] + ".json")
        if os.path.exists(p):
            rows.append(dict(model=tag, rank=int(r["rank"]), seed=int(r["seed"]), run_id=r["run_id"], **one_run(p)))
    if not rows:
        print(f"[{tag}] không khớp file deltaw nào")
        return None
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for m in ("p", "x"):
        ap.add_argument(f"--csv_{m}"); ap.add_argument(f"--lr_map_{m}"); ap.add_argument(f"--deltaw_{m}")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    parts = []
    for m, tag in (("p", "PhoBERT"), ("x", "XLM-R")):
        if getattr(a, f"csv_{m}"):
            d = collect(getattr(a, f"csv_{m}"), getattr(a, f"lr_map_{m}"), getattr(a, f"deltaw_{m}"), tag)
            if d is not None:
                parts.append(d)
    if not parts:
        print("Không có dữ liệu ΔW để tổng hợp."); return
    allr = pd.concat(parts)
    allr["use_ratio"] = allr["eff_rank"] / allr["rank"]
    s = allr.groupby(["model", "rank"])[["eff_rank", "use_ratio", "stable_rank", "fro_norm"]].mean().round(4)
    print(s.to_string())
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        s.reset_index().to_csv(a.out, index=False)


if __name__ == "__main__":
    main()

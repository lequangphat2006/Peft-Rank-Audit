"""In header + 2 dong dau cua moi file tuning de biet schema."""
import os, csv, json

FILES = [
    "results/tuning/tuning_gpu0.csv",
    "results/tuning/tuning_gpu1.csv",
    "results/tuning/tuning_ext_gpu0.csv",
    "results/tuning/tuning_ext_gpu1.csv",
    "results/tuning/tuning_full.csv",
    "results/tuning/tuning_seed400.csv",
    "results/tuning/tuning_5e5.csv",
    "results/ext/phobert/tune_all.csv",
    "results/ext/phobert/tune_all_v2.csv",
    "results/ext/phobert/tune_ctrl_1e4.csv",
    "results/v4_phobert/V4_tune_gpu0.csv",
    "results/v4_phobert/V4_tune_gpu1.csv",
]

def show_csv(path):
    if not os.path.exists(path):
        print("MISSING: " + path); return
    with open(path, encoding="utf-8") as fh:
        r = csv.reader(fh)
        header = next(r, None)
        rows = [next(r, None) for _ in range(2)]
    print("\n" + path)
    print("  cols: " + str(header))
    print("  rows: " + str(len(list(csv.DictReader(open(path, encoding="utf-8"))))))
    for row in rows:
        if row: print("  " + str(row))

def show_json(path):
    if not os.path.exists(path):
        print("MISSING: " + path); return
    print("\n" + path)
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    print("  " + json.dumps(d, indent=2)[:800])

def main():
    print("="*72); print("CSV HEADERS"); print("="*72)
    for f in FILES: show_csv(f)
    print("\n" + "="*72); print("JSON LR MAP"); print("="*72)
    for f in ["results/ext/phobert/lr_map_best.json",
              "results/ext/phobert/tuning_decision.json",
              "results/ext/phobert/lr_control_table.json"]:
        show_json(f)

if __name__ == "__main__":
    main()

"""Tests for run_pilot.py v2 (CPU, tiny random XLM-R, no network).  Run: python -m pytest code/test_run_pilot.py -v"""
import csv, json, os, subprocess, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_pilot as rp  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaForSequenceClassification
    root = tmp_path_factory.mktemp("tiny")
    words = [f"w{i}" for i in range(60)]
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, **{w: i + 4 for i, w in enumerate(words)}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                   unk_token="<unk>")
    mdir = root / "model"; fast.save_pretrained(mdir)
    cfg = XLMRobertaConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=12, num_attention_heads=2,
                           intermediate_size=64, max_position_embeddings=64, pad_token_id=1, num_labels=3)
    XLMRobertaForSequenceClassification(cfg).save_pretrained(mdir)
    rng = np.random.default_rng(0)
    d = root / "data"; d.mkdir()
    for name, n in (("train", 96), ("calibration", 30), ("test", 30)):
        with open(d / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["text", "label"])
            for _ in range(n):
                y = int(rng.integers(0, 3)); w.writerow([" ".join(rng.choice(words[y * 20:(y + 1) * 20 + 5], 6)), y])
    return str(mdir), str(d)


def cli(model, data, out, *extra):
    cmd = [sys.executable, os.path.join(HERE, "run_pilot.py"), "--data_dir", data, "--model", model, "--out", str(out),
           "--device", "cpu", "--batch_size", "16", "--max_len", "32", *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def jobs_for(*argv):
    r = subprocess.run([sys.executable, os.path.join(HERE, "run_pilot.py"), "--data_dir", "x", "--dry_run", "--out",
                        os.path.join(HERE, "_nonexistent.csv"), *argv], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return [l for l in r.stdout.splitlines()[1:] if l.strip()]


def test_global_sweep_job_count_and_shards_partition():
    base = ["--sweep_global", "--ranks", "1", "4", "16", "32", "--epochs_list", "3", "6",
            "--seeds", *map(str, range(200, 208)), "--cell", "R1"]
    allj = jobs_for(*base)
    assert len(allj) == 4 * 2 * 8 == 64 and len(set(allj)) == 64
    s0, s1 = jobs_for(*base, "--shard", "0/2"), jobs_for(*base, "--shard", "1/2")
    assert len(s0) == len(s1) == 32 and set(s0) | set(s1) == set(allj) and not set(s0) & set(s1)
    # split by seed: each shard holds COMPLETE designs (all ranks x budgets) for its seeds
    for sh in (s0, s1):
        seeds = {j.split("__")[1] for j in sh}
        assert len(seeds) == 4 and all(sum(f"__{sd}__" in j for j in sh) == 8 for sd in seeds)
    assert sum("__e6__" in j for j in s0) == sum("__e6__" in j for j in s1) == 16
    # fewer seeds than shards -> fall back to job index, still a partition
    a, b = jobs_for(*base[:9], "--seeds", "200", "--cell", "R1", "--shard", "0/2"), jobs_for(*base[:9], "--seeds", "200", "--cell", "R1", "--shard", "1/2")
    assert len(a) + len(b) == 8 and not set(a) & set(b)


def test_run_id_distinguishes_budgets_and_lr():
    a = rp.make_run_id("R1", 200, "global", -1, 4, 3, 2e-4)
    assert a != rp.make_run_id("R1", 200, "global", -1, 4, 6, 2e-4)
    assert a != rp.make_run_id("R1", 200, "global", -1, 4, 3, 5e-4)
    assert ":" not in rp.make_run_id("R1", 1, "type:attention.self.key", -1, 1, 3, 2e-4)


def test_ranks_for_global_group_type():
    lr, tr = rp.ranks_for("global", -1, 32, 8); assert lr == [32] * 12 and tr is None
    lr, tr = rp.ranks_for("group:middle", -1, 1, 8); assert lr[4:8] == [1] * 4 and lr[:4] == [8] * 4
    lr, tr = rp.ranks_for("type:intermediate.dense", -1, 32, 8); assert lr == [8] * 12 and tr == {"intermediate.dense": 32}


def test_end_to_end_resume_artifacts_and_param_scaling(tiny, tmp_path):
    model, data = tiny; out = tmp_path / "res.csv"
    args = ["--sweep_global", "--ranks", "1", "32", "--epochs_list", "1", "2", "--seeds", "7", "--cell", "T",
            "--precision", "fp32", "--epoch_eval", "both"]
    r = cli(model, data, out, *args); assert r.returncode == 0, r.stdout + r.stderr
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 4
    assert set(CSV := rp.CSV_COLUMNS) >= set(rows[0].keys())
    p = {(int(x["rank"]), int(x["epochs"])): x for x in rows}
    assert int(p[(32, 1)]["lora_params"]) == 32 * int(p[(1, 1)]["lora_params"])          # global rank scales exactly
    for x in rows:
        assert x["cal_nll"] and x["nll"] and x["precision"] == "fp32" and x["f1_per_class"].count(";") == 2
        z = np.load(tmp_path / "logits" / (x["run_id"] + ".npz"))
        assert z["eval_logits"].shape == (30, 3) and z["cal_logits"].shape == (30, 3) and len(z["eval_gold"]) == 30
        ew = json.load(open(tmp_path / "epochwise" / (x["run_id"] + ".json")))["epochs"]
        assert len(ew) == int(x["epochs"]) and "eval_nll" in ew[0] and "cal_nll" in ew[0]
    # resume: nothing new is run, no duplicate rows
    r = cli(model, data, out, *args); assert "0 to run" in r.stdout
    assert len(list(csv.DictReader(open(out, encoding="utf-8")))) == 4
    # meta file written
    assert (tmp_path / "res.csv.meta.jsonl").exists()


def test_class_weight_and_single_config(tiny, tmp_path):
    model, data = tiny; out = tmp_path / "cw.csv"
    r = cli(model, data, out, "--config", "type", "--module_type", "attention.self.key", "--rank", "4", "--seed", "1",
            "--precision", "fp32", "--class_weight", "balanced", "--epochs", "1")
    assert r.returncode == 0, r.stdout + r.stderr
    row = list(csv.DictReader(open(out, encoding="utf-8")))[0]
    assert row["class_weight"] == "balanced" and row["config"] == "type:attention.self.key"


def test_v1_file_is_refused(tiny, tmp_path):
    model, data = tiny; out = tmp_path / "old.csv"
    out.write_text("cell,seed,config,layer,rank,macro_f1\nC1,100,baseline,-1,8,0.8\n", encoding="utf-8")
    r = cli(model, data, out, "--config", "baseline", "--epochs", "1", "--precision", "fp32")
    assert r.returncode != 0 and "run_id" in (r.stdout + r.stderr)

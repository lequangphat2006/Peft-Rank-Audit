"""Tests for run_pilot.py v3 features (CPU, tiny random XLM-R, no network).

Covers: rank 8 default, --method {lora,head,full}, --alpha_mode {2r,fixed:N,rslora},
--data_fraction, delta_w_stats, run_id suffixes, end-to-end artifacts.

Run: python -m pytest code/test_run_pilot_v3.py -v
"""
import csv, json, os, subprocess, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_pilot as rp
import rankcap


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaForSequenceClassification
    root = tmp_path_factory.mktemp("tiny_v3")
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
    cmd = [sys.executable, os.path.join(HERE, "run_pilot.py"),
           "--data_dir", data, "--model", model, "--out", str(out),
           "--device", "cpu", "--batch_size", "16", "--max_len", "32", *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def _pick(v, key="default"):
    """Handle peft fields that may be scalar or dict-of-adapter."""
    return v.get(key, next(iter(v.values()))) if isinstance(v, dict) else v


def _first_lora_layer(peft_model):
    from peft.tuners.lora import Linear as LoraLinear
    for name, mod in peft_model.named_modules():
        if isinstance(mod, LoraLinear):
            return mod
    return None


# --- 1. run_id suffixes only for non-defaults ---------------------------------------------
def test_run_id_v3_suffixes():
    base = rp.make_run_id("R1", 200, "global", -1, 8, 3, 2e-4)
    assert "__m" not in base and "__a" not in base and "__df" not in base
    assert "__mhead" in rp.make_run_id("R1", 200, "global", -1, 8, 3, 2e-4, method="head")
    assert "__mfull" in rp.make_run_id("R1", 200, "global", -1, 8, 3, 2e-4, method="full")
    assert "__afixed16" in rp.make_run_id("R1", 200, "global", -1, 8, 3, 2e-4, alpha_mode="fixed:16")
    assert "__arslora" in rp.make_run_id("R1", 200, "global", -1, 8, 3, 2e-4, alpha_mode="rslora")
    assert "__df0.2" in rp.make_run_id("R1", 200, "global", -1, 8, 3, 2e-4, data_fraction=0.2)
    assert "__df0.5" in rp.make_run_id("R1", 200, "global", -1, 8, 3, 2e-4, data_fraction=0.5)


# --- 2. rank 8 default + param scaling ----------------------------------------------------
def test_rank_8_in_default_sweep_and_scaling(tiny, tmp_path):
    model, data = tiny; out = tmp_path / "r8.csv"
    r = cli(model, data, out,
            "--sweep_global", "--ranks", "1", "4", "8", "16", "32",
            "--epochs_list", "1", "--seeds", "7",
            "--cell", "T8", "--precision", "fp32")
    assert r.returncode == 0, r.stdout + r.stderr
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 5, f"expected 5 ranks, got {len(rows)}"
    by_rank = {int(x["rank"]): int(x["lora_params"]) for x in rows}
    assert set(by_rank.keys()) == {1, 4, 8, 16, 32}
    # LoRA params scale linearly with global rank
    assert by_rank[1] * 32 == by_rank[32]
    assert by_rank[8] * 4 == by_rank[32]
    assert by_rank[16] * 2 == by_rank[32]


# --- 3. alpha_mode scaling factors (unit, in-process) ------------------------------------
def test_alpha_mode_scaling_2r(tiny):
    from peft import get_peft_model
    from transformers import AutoModelForSequenceClassification
    model_dir, _ = tiny
    base = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
    cfg = rankcap.lora_config(base, [8] * 12, base_r=8, alpha_mode="2r")
    m = get_peft_model(base, cfg)
    mod = _first_lora_layer(m)
    assert mod is not None
    r = _pick(mod.r); a = _pick(mod.lora_alpha); s = _pick(mod.scaling)
    assert r == 8 and a == 16, f"2r: r={r} alpha={a} (expected 8, 16)"
    assert abs(s - 2.0) < 1e-6, f"2r: scaling={s} (expected 2.0)"


def test_alpha_mode_scaling_fixed_16(tiny):
    from peft import get_peft_model
    from transformers import AutoModelForSequenceClassification
    model_dir, _ = tiny
    base = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
    cfg = rankcap.lora_config(base, [32] * 12, base_r=32, alpha_mode="fixed:16")
    m = get_peft_model(base, cfg)
    mod = _first_lora_layer(m)
    r = _pick(mod.r); a = _pick(mod.lora_alpha); s = _pick(mod.scaling)
    assert r == 32 and a == 16, f"fixed:16: r={r} alpha={a} (expected 32, 16)"
    assert abs(s - 0.5) < 1e-6, f"fixed:16: scaling={s} (expected 0.5)"


def test_alpha_mode_scaling_rslora(tiny):
    from peft import get_peft_model
    from transformers import AutoModelForSequenceClassification
    model_dir, _ = tiny
    base = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
    cfg = rankcap.lora_config(base, [8] * 12, base_r=8, alpha_mode="rslora")
    m = get_peft_model(base, cfg)
    mod = _first_lora_layer(m)
    r = _pick(mod.r); a = _pick(mod.lora_alpha); s = _pick(mod.scaling)
    assert r == 8 and a == 16
    expected = 16.0 / np.sqrt(8.0)
    assert abs(s - expected) < 1e-5, f"rslora: scaling={s} (expected {expected})"
    # use_rslora flag must be set on config
    assert getattr(m.peft_config["default"], "use_rslora", False) is True


# --- 4. delta_w_stats (unit) --------------------------------------------------------------
def test_delta_w_stats_basic(tiny):
    from peft import get_peft_model
    from transformers import AutoModelForSequenceClassification
    model_dir, _ = tiny
    base = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
    cfg = rankcap.lora_config(base, [4] * 12, base_r=4, alpha_mode="2r")
    m = get_peft_model(base, cfg)
    stats = rp.delta_w_stats(m)
    assert len(stats) > 0, "delta_w_stats returned empty dict"
    for name, s in stats.items():
        assert s["rank"] == 4, f"{name}: rank {s['rank']} != 4"
        assert s["effective_rank"] > 0
        assert s["stable_rank"] > 0
        assert s["fro_norm"] > 0
        assert len(s["singular_values"]) == 4


def test_delta_w_stats_rank_scales(tiny):
    """Higher rank -> more singular values. Reload base each time to avoid double-PEFT warning."""
    from peft import get_peft_model
    from transformers import AutoModelForSequenceClassification
    model_dir, _ = tiny
    for rank in (2, 8):
        base = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
        cfg = rankcap.lora_config(base, [rank] * 12, base_r=rank, alpha_mode="2r")
        m = get_peft_model(base, cfg)
        stats = rp.delta_w_stats(m)
        assert len(stats) > 0
        for s in stats.values():
            assert s["rank"] == rank
            assert len(s["singular_values"]) == rank


# --- 5. --method head: only classifier trainable ------------------------------------------
def test_method_head(tiny, tmp_path):
    model, data = tiny; out = tmp_path / "head.csv"
    r = cli(model, data, out, "--method", "head",
            "--config", "global", "--rank", "8", "--epochs", "1", "--seed", "1",
            "--precision", "fp32")
    assert r.returncode == 0, r.stdout + r.stderr
    row = list(csv.DictReader(open(out, encoding="utf-8")))[0]
    assert row["method"] == "head"
    assert "__mhead" in row["run_id"]
    n = int(row["lora_params"])
    # Kiểm tra tính chất thay vì hardcode kiến trúc:
    #   head-only: ít tham số, chỉ classifier
    #   tiny XLM-R (hidden=32, K=3) dùng 2 lớp: dense(32,32)+out_proj(32,3)
    #   Công thức XLM-R: hidden^2 + hidden + hidden*K + K = 1155
    #   Công thức BERT  : hidden*K + K = 99
    expected_xlmr = 32 * 32 + 32 + 32 * 3 + 3   # 1155
    expected_bert = 32 * 3 + 3                  # 99
    assert 0 < n < 10000, f"head: n={n} ngoài (0, 10000)"
    assert n in (expected_xlmr, expected_bert), \
        f"head: n={n}, expected {expected_xlmr} (XLM-R) hoặc {expected_bert} (BERT)"


# --- 6. --method full: many trainable params, no LoRA ------------------------------------
def test_method_full(tiny, tmp_path):
    model, data = tiny; out = tmp_path / "full.csv"
    r = cli(model, data, out, "--method", "full",
            "--config", "global", "--rank", "8", "--epochs", "1", "--seed", "1",
            "--precision", "fp32")
    assert r.returncode == 0, r.stdout + r.stderr
    row = list(csv.DictReader(open(out, encoding="utf-8")))[0]
    assert row["method"] == "full"
    # tiny XLM-R có hidden=32, 12 layers -> ~100k+ params
    assert int(row["lora_params"]) > 10000, f"full: {row['lora_params']} too few"


# --- 7. --data_fraction: subset + suffix + n_train column --------------------------------
def test_data_fraction(tiny, tmp_path):
    model, data = tiny; out = tmp_path / "df.csv"
    r = cli(model, data, out, "--config", "global", "--rank", "8", "--epochs", "1", "--seed", "1",
            "--precision", "fp32", "--data_fraction", "0.5")
    assert r.returncode == 0, r.stdout + r.stderr
    row = list(csv.DictReader(open(out, encoding="utf-8")))[0]
    assert "__df0.5" in row["run_id"]
    assert float(row["data_fraction"]) == 0.5
    # 96 train rows -> stratified 0.5 -> ~48; allow some slack for rounding
    assert 40 <= int(row["n_train"]) <= 55, f"n_train={row['n_train']} outside [40,55]"


# --- 8. end-to-end with v3 features; verify artifacts ------------------------------------
def test_v3_end_to_end_artifacts(tiny, tmp_path):
    model, data = tiny; out = tmp_path / "v3e2e.csv"
    r = cli(model, data, out,
            "--sweep_global", "--ranks", "4", "8",
            "--epochs_list", "1", "--seeds", "7",
            "--cell", "T3", "--precision", "fp32",
            "--data_fraction", "0.8", "--epoch_eval", "both")
    assert r.returncode == 0, r.stdout + r.stderr
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 2
    for x in rows:
        assert x["method"] == "lora"
        assert x["alpha_mode"] == "2r"
        assert abs(float(x["data_fraction"]) - 0.8) < 1e-9
        # deltaw artifact
        dw_path = tmp_path / "deltaw" / (x["run_id"] + ".json")
        assert dw_path.exists(), f"missing deltaw artifact: {dw_path}"
        dw = json.load(open(dw_path))
        assert dw["run_id"] == x["run_id"]
        assert isinstance(dw["layers"], dict) and len(dw["layers"]) > 0
        # logits + epochwise still present
        assert (tmp_path / "logits" / (x["run_id"] + ".npz")).exists()
        assert (tmp_path / "epochwise" / (x["run_id"] + ".json")).exists()
    # meta.jsonl should record v3 fields
    meta = [json.loads(l) for l in open(str(out) + ".meta.jsonl", encoding="utf-8")]
    assert any(m.get("method") == "lora" and m.get("alpha_mode") == "2r" for m in meta)
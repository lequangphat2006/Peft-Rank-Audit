"""Test run_adalora.py tren model XLM-R tieu hon (random init), khong can tai gi tu HF Hub."""
import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_adalora as ra  # noqa: E402


def _tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    for w in "a b c d e f g h i j".split():
        vocab[w] = len(vocab)
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", unk_token="<unk>",
                                   bos_token="<s>", eos_token="</s>")


def _tiny_model(num_labels):
    from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification
    cfg = XLMRobertaConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=64, max_position_embeddings=64, num_labels=num_labels,
                           pad_token_id=0)
    return XLMRobertaForSequenceClassification(cfg)


def _write_split(path, n, seed):
    import random
    rnd = random.Random(seed)
    words = "a b c d e f g h i j".split()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["text", "label"])
        for _ in range(n):
            lab = rnd.randrange(3)
            w.writerow([" ".join(rnd.choice(words) for _ in range(rnd.randint(3, 8))), lab])


@pytest.fixture
def tiny_env(tmp_path, monkeypatch):
    for name, n in (("train.csv", 96), ("calibration.csv", 24), ("test.csv", 24)):
        _write_split(tmp_path / name, n, seed=hash(name) % 1000)
    tok = _tiny_tokenizer()

    class _Tok:
        @staticmethod
        def from_pretrained(*a, **k): return tok

    class _Model:
        @staticmethod
        def from_pretrained(name, num_labels=3): return _tiny_model(num_labels)

    monkeypatch.setattr(ra, "AutoTokenizer", _Tok)
    monkeypatch.setattr(ra, "AutoModelForSequenceClassification", _Model)
    return tmp_path


def _argv(env, out, extra=()):
    return ["--data_dir", str(env), "--out", str(out), "--seeds", "300", "--epochs", "2",
            "--init_rank", "4", "--target_rank", "2", "--batch_size", "8", "--deltaT", "2",
            "--tinit_frac", "0.1", "--tfinal_frac", "0.4", "--precision", "fp32", "--device", "cpu", *extra]


def test_rejects_init_not_greater_than_target(tiny_env, tmp_path):
    with pytest.raises(SystemExit):
        ra.parse_args(["--data_dir", "x", "--out", "y", "--init_rank", "8", "--target_rank", "16"])


def test_schedule_steps_peft_semantics():
    assert ra.schedule_steps(2400, 0.1, 0.4, 50) == (240, 960)
    with pytest.raises(ValueError):
        ra.schedule_steps(2400, 0.1, 0.9, 50)  # cua so cat tia qua hep


def test_run_id_unique_and_stable():
    a = ra.make_run_id("ADA", 300, 12, 8, 6, 2e-4, 0.1, 0.4)
    b = ra.make_run_id("ADA", 301, 12, 8, 6, 2e-4, 0.1, 0.4)
    assert a != b and a == ra.make_run_id("ADA", 300, 12, 8, 6, 2e-4, 0.1, 0.4)


def test_end_to_end_prunes_to_budget_and_resumes(tiny_env, tmp_path):
    out = tmp_path / "res" / "adalora.csv"
    ra.main(_argv(tiny_env, out))
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 1
    r = rows[0]
    assert r["method"] == "adalora" and r["config"] == "adalora" and r["rank"] == "2"
    assert 0.0 <= float(r["macro_f1"]) <= 1.0 and float(r["nll"]) > 0
    # 2 layer x 6 module = 12 module, target_r = 2 -> ngan sach 24
    assert int(r["target_budget"]) == 24
    assert int(r["final_rank_sum"]) <= 24 + 12  # sai so do hoa (ties) khong vuot qua 1 triplet/module
    assert int(r["final_rank_max"]) <= 4 and int(r["final_rank_min"]) >= 0
    # chi 12 module duoc gan, khong dinh classifier.dense
    rk = json.load(open(tmp_path / "res" / "adalora_rank" / (r["run_id"] + ".json")))["ranks"]
    assert len(rk) == 12 and all("classifier" not in k for k in rk)
    assert (tmp_path / "res" / "logits" / (r["run_id"] + ".npz")).exists()
    # chay lai: run_id da xong -> bo qua
    ra.main(_argv(tiny_env, out))
    assert len(list(csv.DictReader(open(out, encoding="utf-8")))) == 1


def test_loss_includes_orth_reg(tiny_env):
    """Loss tu model phai lon hon CE thuan (PEFT cong orthogonal regularization khi truyen labels)."""
    import torch
    import torch.nn.functional as F
    args = ra.parse_args(_argv(tiny_env, tiny_env / "o.csv"))
    peft_model, _, _ = ra.build_adalora_model(args, 3, 300, total_steps=16, tinit=1, tfinal=6)
    ids = torch.randint(4, 14, (8, 6)); mask = torch.ones_like(ids); y = torch.randint(0, 3, (8,))
    peft_model.eval()
    out = peft_model(input_ids=ids, attention_mask=mask, labels=y)
    ce = F.cross_entropy(out.logits.float(), y)
    assert float(out.loss) > float(ce) + 1e-6

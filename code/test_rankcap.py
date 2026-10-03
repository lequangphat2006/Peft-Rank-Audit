import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""CPU tests (no downloads): python -m pytest -q  or  python test_rankcap.py"""
import numpy as np, pandas as pd, torch
import rankcap, analyze_pilot as ap


def tiny_model():
    from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification
    cfg = XLMRobertaConfig(vocab_size=120, hidden_size=32, num_hidden_layers=12, num_attention_heads=2,
                           intermediate_size=64, max_position_embeddings=140, num_labels=3)
    return XLMRobertaForSequenceClassification(cfg)


def test_allocate_ranks():
    rng = np.random.default_rng(0)
    for _ in range(200):
        sig = rng.uniform(0, 1, 12)
        r = rankcap.allocate_ranks(sig, 96)
        assert r.sum() == 96 and r.min() >= 1 and r.max() <= 32
    # monotone: higher sigma never gets lower rank
    sig = np.linspace(0, 1, 12)
    r = rankcap.allocate_ranks(sig, 96)
    assert (np.diff(r) >= 0).all()
    # constant sigma -> uniform 8
    assert (rankcap.allocate_ranks(np.full(12, .5), 96) == 8).all()


def test_controls_same_budget():
    c = rankcap.make_controls(np.random.rand(12), 96, np.random.default_rng(1))
    assert all(v.sum() == 96 for v in c.values())


def test_patterns_and_budget():
    m = tiny_model(); names = rankcap.module_names(m)
    assert len(names) == 72 and not any(n.startswith("classifier") for n in names)
    from peft import get_peft_model
    pm = {}
    for tag, lr in {"uniform": [8] * 12, "dyn": list(rankcap.allocate_ranks(np.linspace(0, 1, 12), 96))}.items():
        model = get_peft_model(tiny_model(), rankcap.lora_config(tiny_model(), lr))
        pm[tag] = (model, rankcap.count_lora_params(model))
    assert pm["uniform"][1] == pm["dyn"][1], "same rank sum must mean same LoRA params at layer level"
    model = pm["dyn"][0]; lr = list(rankcap.allocate_ranks(np.linspace(0, 1, 12), 96))
    seen = 0
    for n, mod in model.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            layer = int(n.split("encoder.layer.")[1].split(".")[0])
            assert mod.lora_A["default"].weight.shape[0] == lr[layer]
            assert abs(mod.scaling["default"] - 2.0) < 1e-9, "alpha/r must stay 2"
            assert "classifier" not in n
            seen += 1
    assert seen == 72


def test_type_sweep():
    m = tiny_model()
    t, rp, ap_ = rankcap.build_patterns(m, [8] * 12, type_ranks={"intermediate.dense": 32})
    assert all(rp[n] == 32 for n in t if n.endswith("intermediate.dense"))
    assert all(rp[n] == 8 for n in t if n.endswith("attention.self.query"))


def test_imbalance():
    for rho in (1, 5, 20):
        n = rankcap.imbalanced_counts(3, 3000, rho)
        assert n.sum() == 3000 and abs(n.max() / n.min() - rho) / rho < 0.02
    assert rankcap.max_feasible_N([5000, 5000, 400]) == 1200
    assert rankcap.feasible_imbalance([5000, 5000, 400], 3000) == {1: False, 5: True, 20: True}


def synth(sigma_lr, sigma_eps, S=5, L=12, seed=0):
    rng = np.random.default_rng(seed); u = rng.normal(0, sigma_lr, L); rows = []
    for s in range(S):
        base = 0.80 + rng.normal(0, 0.01)
        rows.append(["C1", 100 + s, "baseline", -1, 8, base, 0, 0])
        for m in range(L):
            for r in (1, 32):
                rows.append(["C1", 100 + s, "layer", m, r, base + (-u[m] if r == 1 else 0) + rng.normal(0, sigma_eps), 0, 0])
    return pd.DataFrame(rows, columns=["cell", "seed", "config", "layer", "rank", "macro_f1", "lora_params", "seconds"])


def test_analysis_detects_signal_and_noise():
    strong = ap.analyse(synth(0.02, 0.004))["by_rank"][1]
    assert strong["rho"] > 0.8 and strong["perm_p"] < 0.01
    null = [ap.analyse(synth(0.0, 0.01, seed=i))["by_rank"][1] for i in range(20)]
    assert np.mean([x["perm_p"] < 0.05 for x in null]) < 0.25   # ~5% expected, loose bound for 20 draws
    assert np.mean([x["rho"] for x in null]) < 0.35


def test_power_sim_monotone():
    lo = ap.simulate_power(0.002, 0.01, 5, nsim=150)["power_H1"]
    hi = ap.simulate_power(0.02, 0.01, 5, nsim=150)["power_H1"]
    assert hi > lo and hi > 0.8


def test_end_to_end_tiny_training():
    import run_pilot
    from peft import get_peft_model
    torch.manual_seed(0)
    X = torch.randint(3, 120, (64, 12)); y = (X[:, 0] % 3)
    data = list(zip(X, torch.ones_like(X), y))
    dl = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)
    model = get_peft_model(tiny_model(), rankcap.lora_config(tiny_model(), [8] * 12))
    f1 = run_pilot.train_eval(model, dl, dl, epochs=2, lr=2e-4, device="cpu")
    assert 0.0 <= f1 <= 1.0


def test_dynamic_padding_loader():
    """make_loader pads each batch to its own longest sequence, keeps labels aligned, tolerates a BOM file."""
    import run_pilot
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    texts = ["tot"] * 8 + ["giang day rat nhiet tinh va gan gui voi sinh vien"] * 8
    tk = Tokenizer(models.WordLevel(unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(texts, trainers.WordLevelTrainer(special_tokens=["<s>", "<pad>", "</s>", "<unk>"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
    labels = [i % 3 for i in range(len(texts))]
    dl = run_pilot.make_loader(tok, texts, labels, 128, 4, False, 0)
    lens = [ids.shape[1] for ids, mask, y in dl]
    assert min(lens) < max(lens) < 128, "batches should be padded to their own max length"
    ids, mask, y = next(iter(dl))
    assert ids.shape == mask.shape and y.tolist() == labels[:4]
    import tempfile, os
    f = os.path.join(tempfile.mkdtemp(), "t.csv")
    open(f, "w", encoding="utf-8-sig").write("text,label\nxin chao,2\n")
    assert run_pilot.load_csv(f) == (["xin chao"], [2])


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-q"]))

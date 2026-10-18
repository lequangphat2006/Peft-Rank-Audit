# How Much Does Global LoRA Rank Matter? A Preregistered Equivalence and Calibration Audit on XLM-R/UIT-VSFC

**Le Quang Phat, Dinh Thi Hong Huyen**
*Faculty of Information Technology, Quy Nhon University, Vietnam*

## Abstract
Adaptive rank allocation methods (e.g., AdaLoRA, IGU-LoRA) assume that different components of a neural network have distinct "capacity needs" and allocate LoRA ranks accordingly. However, whether increasing the global LoRA rank actually improves downstream performance remains an open question, particularly when the learning rate (LR) is tuned fairly. In this paper, we present a preregistered equivalence and calibration audit on XLM-R and UIT-VSFC. Using TOST equivalence testing (δ = 0.01) and Holm correction, we find that increasing the global rank from 1 to 32 does not improve Macro-F1. While the effect is statistically equivalent at the seed-level variance, a sample-level bootstrap reveals that test-set noise still poses a boundary to absolute equivalence (90% CI [-0.013, +0.006]). Nonetheless, the rank significantly degrades calibration (NLL increases). Crucially, a post-hoc analysis reveals that the observed rank-by-epoch interaction (H3) is a confounder: it reverses direction when LR is fixed, indicating that "rank effects" are highly sensitive to hyperparameter choices. Furthermore, we demonstrate that the effect is model-dependent; while rank 1 suffices for XLM-R, smaller models like PhoBERT benefit from higher ranks. Finally, a simple global LoRA rank 8 with properly tuned LR outperforms the adaptive AdaLoRA baseline. Our results suggest that the complex search for "optimal rank allocation" may be optimizing on noise, and practitioners should prioritize LR tuning over rank scaling.

## 1. Introduction
Parameter-Efficient Fine-Tuning (PEFT), particularly Low-Rank Adaptation (LoRA), has become the standard for adapting large language models due to its efficiency. While vanilla LoRA uses a fixed global rank, recent methods such as AdaLoRA, IGU-LoRA, and StatLoRA dynamically allocate ranks across different modules, assuming that each module possesses a unique and measurable "capacity need." Yet, recent studies suggest that rank-one sufficiency might hold across various settings, challenging the necessity of complex rank allocation. This raises a fundamental question: *Does increasing the global LoRA rank actually improve downstream performance?*

A critical, yet often overlooked, confounder in rank comparison is the learning rate (LR). Because LoRA scales its updates by a ratio related to rank, higher ranks fundamentally alter the effective learning dynamics. If the optimal LR shifts with rank, comparing ranks under a fixed LR severely biases the results. Furthermore, numerical precision (e.g., FP16) and the scaling factor (α) can induce artificial "rank effects" such as model collapse or overconfidence. Thus, a rigorous audit of rank scaling requires an equivalence test where the LR is tuned per rank, and potential confounders are systematically isolated.

In this paper, we conduct a preregistered audit to answer: *How much does global LoRA rank matter for discrimination and calibration when LR is tuned per rank?* Our contributions are fourfold:
1. **Preregistered Equivalence Audit:** Using TOST on 160 runs on XLM-R/UIT-VSFC, we demonstrate that F1 is equivalent from rank 1 to 32 at the seed-level. However, we transparently report that sample-level bootstrap widens the CI to [-0.013, +0.006], meaning test-set noise still poses a boundary to absolute equivalence. Meanwhile, NLL significantly degrades with higher ranks.
2. **The LR Confounder (The Twist):** We show that the apparent rank-by-epoch interaction reverses sign when LR is fixed (from +0.017 to -0.033). This proves that prior "rank effects" may be artifacts of hyperparameter misconfiguration rather than inherent model properties.
3. **Model-Dependence:** Through extensions to PhoBERT, we demonstrate that rank effects are architecture-dependent: while rank 1 suffices for XLM-R, smaller models require higher ranks to achieve optimal F1.
4. **Adaptive Rank Baseline:** We show that AdaLoRA underperforms a simple global rank 8 under matched conditions, suggesting that gradient-based importance scores may be optimizing on noise.

## 2. Related Work
**LoRA and Adaptive Rank Allocation.** Standard LoRA injects trainable rank-decomposition matrices into the model. Building on this, AdaLoRA [Zhang et al., 2023] formulates rank allocation as an SVD-based importance scoring problem, pruning less important singular values. IGU-LoRA [2026] improves this by using integrated gradients to stabilize the importance scores. StatLoRA [Gao et al., 2026] frames rank allocation as a statistical hypothesis testing problem. All these methods assume that a measurable "capacity need" exists and is stable.

**Learning Rate and Rank Sufficiency.** Recent work [Lee et al., 2026] demonstrates that when LR is tuned fairly, many complex LoRA variants perform equivalently to vanilla LoRA. Furthermore, [Rethinking the Rank Threshold, 2026] shows rank-one sufficiency across various axes. Our work differs by using a preregistered equivalence test (TOST) and explicitly isolating the LR confounder via a post-hoc diagnostic.

**Calibration in PEFT.** Neural networks are often overconfident [Guo et al., 2017]. While calibration under distribution shift has been studied, the effect of global LoRA rank on NLL and ECE—especially when LR is tuned—remains under-explored.

## 3. Methodology

### 3.1. Dataset and Preprocessing
We conduct our audit on the UIT-VSFC dataset [Nguyen et al., 2018], a Vietnamese student feedback corpus with three sentiment classes (negative, neutral, positive). The dataset is split into 12,780 training examples, 1,598 calibration examples, and 1,599 test examples. The class distribution is highly imbalanced, with the neutral class comprising only ~4% of the training data. Text is tokenized using the XLM-R tokenizer with a maximum length of 128 and dynamic padding. A near-duplicate analysis (TF-IDF char_wb n-grams, cosine ≥ 0.95) identified 3.75% of the test set as near-duplicates to the training set; post-hoc checks confirmed that removing them does not change the conclusions.

### 3.2. Proposed Methods
**3.2.1. Global LoRA Rank Scaling (The Intervention).** We use `FacebookAI/xlm-roberta-base` and inject LoRA adapters into all 12 transformer layers across 6 target modules (Query, Key, Value, Attention Output, FFN-Intermediate, FFN-Output), totaling 72 modules. We vary the global rank \(r \in \{1, 4, 8, 16, 32\}\), applying the same rank to all 72 modules. The scaling factor is set to \(\alpha = 2r\) (constant \(\alpha/r = 2\)).

**3.2.2. Preregistered Audit Protocol.** The confirmatory analysis is structured around three hypotheses (family H1–H3) using Holm correction (\(m=3\)). H1: The total effect of rank 1→32 on Macro-F1 is equivalent to zero (TOST, \(\delta = 0.01\)). H2: NLL increases with rank (one-sided). H3: The slope of Macro-F1 differs between 3 and 6 epochs (F-test). To ensure fairness, we tune the LR strictly on the calibration split using 3 seeds (400-402) and a grid of \(\{1e-4, 2e-4, 5e-4, 1e-3\}\). 

**3.2.3. Comparison with Adaptive Rank (AdaLoRA).** To compare our simple rank scaling with adaptive methods, we run AdaLoRA [Zhang et al., 2023]. We configure it with `init_r=12` and `target_r=8`, resulting in a final average rank budget of 8 (total budget = 576). We run it with the LR inherited from LoRA rank 8 (1e-4) to ensure a fair starting point.

### 3.3. Secondary Model and Task
To test model dependence, we replicate the sweep on **PhoBERT-base** (a monolingual Vietnamese model) under the same LR tuning protocol. To test task dependence, we run XLM-R on **AG News** (a 4-class English topic classification task).

## 4. Experimental Protocol

### 4.1. Data Splitting and Evaluation Sets
The test split (1,599 examples) is treated as the sole untouched hold-out and is used for all primary evaluations. The calibration split (1,598 examples) is used exclusively for LR selection and temperature scaling. No hyperparameter or model selection uses the test split.

### 4.2. Configuration, Metrics, and Statistical Testing
**Training:** Batch size 32, max length 128, 3 and 6 epochs, mixed precision (FP16), and AdamW optimizer. We report Macro-F1 (discrimination) and unweighted NLL (calibration). 
**Statistical Models:** We fit Ordinary Least Squares (OLS) regressions using `Metric ~ C(seed) + log2(rank)`. Seed is a fixed effect. 
**Post-hoc Diagnostics:** To isolate confounders, we run: (V1) A fixed-LR branch (2e-4 for all ranks) to separate rank effects from LR scheduling. (V5) FP32 validation for collapsed runs. (Alpha) A sensitivity check comparing \(\alpha=2r\) with \(\alpha=16\).

## 5. Experimental Results and Evaluation

### 5.1. In-Domain Equivalence and Calibration (H1, H2)
**H1 (Equivalence):** As shown in **Figure 2a**, Macro-F1 remains flat across all ranks. The TOST procedure confirms equivalence to zero with a 90% CI of \([-0.0084, +0.0016]\), entirely within the bound of \(\delta = 0.01\) (\(p_{holm} = 0.016\)). A test-sample bootstrap (**Figure 3**) widens the CI to \([-0.013, +0.006]\), crossing the lower threshold, which we transparently report as a boundary imposed by test-set noise.

**H2 (Calibration):** Increasing rank significantly degrades calibration. **Figure 2b** shows a clear upward trend in NLL. The total effect of rank 1→32 is +0.0049 (95% CI \([+0.0010, +0.0087]\), \(p_{holm} = 0.014\)). Temperature scaling (**Figure 6**) reduces overall NLL and ECE but does not eliminate the rank-induced gap, confirming that higher ranks induce genuine overconfidence.

### 5.2. The Reversal of Rank × Epoch (H3 & Post-hoc)
The preregistered H3 showed a significant interaction (\(p_{holm} = 0.007\)). However, post-hoc diagnostics (**Figure 5**) exposed a crucial confounder. When LR is fixed at 2e-4 across all ranks, the interaction **reverses sign** (from +0.017 to -0.033). At 3 epochs, higher ranks converge faster (+0.049), but at 6 epochs, the slope flattens (+0.016). This proves that the initial H3 interaction was an artifact of reusing 6-epoch LRs for 3-epoch runs, rendering H3 **not robust** to LR selection.

### 5.3. Adaptive Rank Baseline (AdaLoRA)
AdaLoRA underperforms our simple baselines. As shown in **Figure 11**, AdaLoRA achieves an F1 of ~0.73. Even when we disable its orthogonal regularization (**V6 diagnostic**), F1 only rises to 0.78, still significantly lower than a simple LoRA rank 8 (F1 0.834). **Figure 12** shows that AdaLoRA's final rank allocation is stable across seeds (Kendall \(\tau \approx 0.66\)), suggesting that gradient-based importance scores may be consistently pruning directions that are critical for the downstream task.

### 5.4. Model Dependence (PhoBERT & AG News)
**Figure 8** demonstrates that rank effects are architecture-dependent. While XLM-R exhibits equivalence, PhoBERT (a smaller model) shows a positive rank effect. At 6 epochs, PhoBERT's F1 increases from 0.798 (rank 1) to 0.819 (rank 32), though NLL also degrades. On AG News, the effect is practically flat, indicating task dependence.

## 6. Conclusion
This paper evaluated the effect of global LoRA rank under a preregistered equivalence and calibration audit. We find that rank 1 is sufficient for XLM-R on UIT-VSFC, while increasing rank induces overconfidence (higher NLL). Critically, we demonstrate that observed "rank effects" (such as rank × epoch interactions) are highly sensitive to confounders like LR tuning and scaling factors. Finally, a simple global rank 8 outperforms the complex AdaLoRA baseline. Practitioners should prioritize LR tuning over rank scaling.

**Limitations.** The audit is limited to one primary dataset (UIT-VSFC) and one primary model (XLM-R). The fixed data split means that test-set noise remains a boundary (as seen in the bootstrap CI). Near-duplicate samples (3.75%) were retained to reflect the natural distribution, though a sensitivity check confirmed they do not drive the conclusions.

**Future Work.** Extending this audit to decoder-only LLMs (e.g., Qwen 1.5B) and multilingual tasks would test whether scale or architecture changes the rank ordering.

## References
- Hu et al. (2022) — LoRA. ICLR.
- Zhang et al. (2023) — AdaLoRA. ICLR.
- Lee et al. (2026) — Learning Rate Matters. arXiv:2602.04998.
- Biderman et al. (2024) — LoRA Learns Less and Forgets Less. TMLR.
- Guo et al. (2017) — On calibration of modern neural networks. ICML.
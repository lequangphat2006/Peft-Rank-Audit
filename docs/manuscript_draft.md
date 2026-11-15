# How Much Does Global LoRA Rank Matter? A Preregistered Equivalence and Calibration Audit on Vietnamese Text Classification

**Le Quang Phat, Dinh Thi Hong Huyen**
*Faculty of Information Technology, Quy Nhon University, Vietnam*

## Abstract

Adaptive rank allocation methods for Low-Rank Adaptation (LoRA) assume that model components have distinct "capacity needs" and allocate ranks accordingly. However, whether increasing the global LoRA rank actually improves downstream performance under fair hyperparameter tuning remains highly debated. We present a preregistered equivalence and calibration audit on XLM-R and PhoBERT using the UIT-VSFC dataset, evaluating the global rank from 1 to 32. Using TOST equivalence testing ($\delta = 0.01$), we find that increasing the global rank from 1 to 32 does not improve Macro-F1. While equivalence holds under seed-level variance (90% CI $[-0.008, +0.002]$), a two-tier bootstrap reveals that test-set noise widens the CI to $[-0.013, +0.006]$. Crucially, we expose a hidden confounder: cross-session hardware variance and learning rate (LR) selection. While initial unblocked runs suggested a model-specific rank interaction (E1), a session-blocked design completely nullified this effect ($\Delta = +0.0014, p_{holm} = 0.70$). A random-effects meta-analysis across four independent iterations ($\tau = 0.006$) confirms that the pooled interaction is effectively zero. Furthermore, a simple global LoRA rank 8 with properly tuned LR outperforms the complex AdaLoRA baseline. Our findings indicate that apparent "rank effects" in prior literature are largely artifacts of LR selection and environment variance, suggesting practitioners should prioritize LR tuning and environment control over complex rank scaling.

## 1. Introduction

Parameter-Efficient Fine-Tuning (PEFT), particularly Low-Rank Adaptation (LoRA), has become the standard for adapting large language models due to its efficiency. While vanilla LoRA uses a fixed global rank, recent methods such as AdaLoRA, IGU-LoRA, and StatLoRA dynamically allocate ranks across different modules, assuming that each module possesses a unique and measurable "capacity need." Yet, recent studies suggest that rank-one sufficiency might hold across various settings, challenging the necessity of complex rank allocation. This raises a fundamental question: *Does increasing the global LoRA rank actually improve downstream performance?*

A critical, yet often overlooked, confounder in rank comparison is the learning rate (LR). Because LoRA scales its updates by a ratio related to rank (\(\alpha/r\)), higher ranks fundamentally alter the effective learning dynamics. If the optimal LR shifts with rank, comparing ranks under a fixed LR severely biases the results. Furthermore, we demonstrate that **hardware variance across cloud sessions** (e.g., different Kaggle GPU allocations) introduces a F1 gap of up to 0.02, which is on the same order of magnitude as the entire rank effect reported in prior literature. Thus, a rigorous audit of rank scaling requires an equivalence test where the LR is tuned per rank, and environment variance is strictly controlled via session-blocking.

In this paper, we conduct a preregistered audit to answer: *How much does global LoRA rank matter for discrimination and calibration when LR is tuned per rank and environment variance is controlled?* Our contributions are fourfold:
1. **Preregistered Equivalence Audit:** Using 160 confirmatory runs, we demonstrate that F1 is equivalent from rank 1 to 32 at the seed level. However, a two-tier bootstrap reveals that test-set noise widens the CI to $[-0.013, +0.006]$, meaning test-set noise still poses a boundary to absolute equivalence. Meanwhile, NLL significantly degrades with higher ranks.
2. **The LR Confounder:** We show that the apparent rank-by-epoch interaction reverses sign when LR is fixed (from +0.017 to -0.033). This proves that prior "rank effects" may be artifacts of hyperparameter misconfiguration.
3. **Protocol Dependence (Not Model Dependence):** While initial unblocked runs suggested PhoBERT benefits from higher ranks, our session-blocked design (T2) nullified this interaction ($p_{holm} = 0.70$). A meta-analysis across four iterations (Main, T2, T3-Fold2, T3-Fold3) yields a pooled interaction of $+0.0019$ with $\tau = 0.006$, confirming the effect is dominated by environmental noise.
4. **Adaptive Rank Baseline:** We show that AdaLoRA underperforms a simple global rank 8 under matched tuning conditions, suggesting that gradient-based importance scores may be optimizing on noise.

## 2. Related Work

**LoRA and Adaptive Rank Allocation.** Standard LoRA injects trainable rank-decomposition matrices into the model. Building on this, AdaLoRA [Zhang et al., 2023] formulates rank allocation as an SVD-based importance scoring problem. IGU-LoRA [2026] improves this by using integrated gradients, and StatLoRA [Gao et al., 2026] frames rank allocation as statistical hypothesis testing. All these methods assume that a measurable "capacity need" exists.

**Learning Rate and Rank Sufficiency.** Recent work [Lee et al., 2026] demonstrates that when LR is tuned fairly, complex LoRA variants perform equivalently to vanilla LoRA. Furthermore, [Rethinking the Rank Threshold, 2026] shows rank-one sufficiency. Our work differs by using a preregistered equivalence test (TOST), explicitly isolating the LR confounder, and controlling for cross-session hardware variance.

**Calibration in PEFT.** Neural networks are often overconfident [Guo et al., 2017]. While calibration under distribution shift has been studied, the effect of global LoRA rank on NLL and ECE—especially when LR is tuned—remains under-explored.

## 3. Methodology

### 3.1. Dataset and Preprocessing
We conduct our audit on the UIT-VSFC dataset [Nguyen et al., 2018], a Vietnamese student feedback corpus with three sentiment classes. The dataset is split into 12,780 training, 1,598 calibration, and 1,599 test examples. The class distribution is highly imbalanced (neutral class $\approx$ 4%). Near-duplicate analysis (TF-IDF char_wb n-grams, cosine $\ge$ 0.95) identified 3.75% of the test set as near-duplicates; post-hoc checks confirmed that removing them does not change the conclusions.

### 3.2. Global LoRA Rank Scaling (The Intervention)
We use `xlm-roberta-base` and `vinai/phobert-base`, injecting LoRA adapters into all 12 layers across 6 target modules (Query, Key, Value, Attention Output, FFN-Intermediate, FFN-Output), totaling 72 modules. We vary the global rank \(r \in \{1, 4, 8, 16, 32\}\) with the scaling factor set to \(\alpha = 2r\) (constant \(\alpha/r = 2\)).

### 3.3. Preregistered Hypotheses
The confirmatory analysis is structured around three hypotheses (family H1–H3) using Holm correction (\(m=3\)):
- **H1:** The total effect of rank $1 \to 32$ on Macro-F1 is equivalent to zero (TOST, $\delta = 0.01$).
- **H2:** NLL increases with rank (one-sided).
- **H3:** The slope of Macro-F1 differs between 3 and 6 epochs (F-test).

For the extension (EXT-1), we tested:
- **E1:** Model $\times$ rank interaction on Macro-F1.
- **E2:** PhoBERT equivalence to zero ($\delta = 0.01$).
- **E3:** PhoBERT NLL increases with rank.

To ensure fairness, LR is tuned strictly on the calibration split using 3 seeds (400-402) and a grid of \(\{1e-4, 2e-4, 5e-4, 1e-3\}\).

### 3.4. Session-Blocked Design and Meta-Analysis (Post-hoc)
To control for cross-session hardware variance (which introduced a ~0.02 F1 gap), we executed a **session-blocked design (T2)**. PhoBERT and XLM-R runs were interleaved within a single Kaggle session to ensure identical hardware conditions. To further test stability, we ran **T3 (Alternative Data Splits)** using 2 additional stratified folds (Fold 2 and Fold 3) with 4 seeds each. A random-effects meta-analysis (Hartung-Knapp) was applied across the Main, T2, T3-F2, and T3-F3 iterations.

## 4. Experimental Protocol

### 4.1. Data Splitting and Evaluation Sets
The test split is the sole untouched hold-out for all primary evaluations. The calibration split is used exclusively for LR selection and temperature scaling. No hyperparameter selection uses the test split.

### 4.2. Configuration, Metrics, and Statistical Testing
**Training:** Batch size 32, max length 128, 3 and 6 epochs, mixed precision (FP16), AdamW optimizer. We report Macro-F1 and unweighted NLL.
**Statistical Models:** We fit OLS regressions using `Metric ~ C(seed) + log2(rank)`. We also run a two-tier bootstrap (resampling both test sentences and seeds) to quantify test-set noise.

## 5. Experimental Results

### 5.1. In-Domain Equivalence and Calibration (H1, H2)
**H1 (Equivalence):** Macro-F1 remains flat across all ranks. TOST confirms equivalence to zero with a 90% CI of \([-0.0084, +0.0016]\) (\(p_{holm} = 0.016\)). A two-tier bootstrap widens the CI to \([-0.013, +0.006]\), crossing the lower threshold, which we report as a boundary imposed by test-set noise.
**H2 (Calibration):** NLL significantly degrades with rank. The total effect is $+0.0049$ (95% CI $[+0.0010, +0.0087]$, \(p_{holm} = 0.014\)). Temperature scaling reduces overall NLL but does not eliminate the rank-induced gap.

### 5.2. The Reversal of Rank $\times$ Epoch (H3)
The preregistered H3 showed a significant interaction (\(p_{holm} = 0.007\)). However, post-hoc diagnostics exposed a confounder. When LR is fixed at 2e-4 across all ranks, the interaction **reverses sign** (from +0.017 to -0.033). This proves that the initial H3 interaction was an artifact of reusing 6-epoch LRs for 3-epoch runs. H3 is **not robust**.

### 5.3. Protocol Dependence and Environment Variance (E1, T2, T3)
**The Illusion of Model Dependence:** In our initial unblocked runs, PhoBERT appeared to benefit from higher ranks (F1 $0.798 \to 0.819$), yielding a suggestive model $\times$ rank interaction (E1: $\Delta = +0.0073$, \(p_{holm} = 0.065\)). 
**Session-Blocked Design (T2):** To isolate environment variance, we interleaved models in a single session. The interaction completely vanished (E1: $\Delta = +0.0014$, 95% CI $[-0.006, +0.009]$, \(p_{holm} = 0.70\)). Both models showed equivalent performance across ranks. Bootstrap confirmed the CI contains zero.
**Meta-Analysis (T3):** Across four iterations (Main, T2, T3-F2, T3-F3), the pooled E1 effect was $+0.0019$ (95% CI $[-0.0097, +0.0135]$). The between-iteration variance (\(\tau = 0.0060\)) exceeded the pooled effect size, confirming that E1 is dominated by environmental noise. Under a fixed-LR protocol, the interaction reversed sign again ($\Delta = -0.0066$).

### 5.4. Mechanistic Analysis: Effective Rank (Delta W)
To understand why increasing rank does not improve Macro-F1, we analyzed the effective rank of the LoRA $\Delta W$ matrices. As nominal rank increased from 1 to 32, the effective rank grew linearly from 1.0 to approximately 19.0 (PhoBERT) and 17.1 (XLM-R). Despite utilizing this extra capacity, Macro-F1 did not improve. This suggests the task's intrinsic dimensionality is saturated at rank 4, and the additional rank is used to memorize noise rather than task-relevant features.

### 5.5. Adaptive Rank Baseline (AdaLoRA)
AdaLoRA underperforms our simple baselines. It achieves an F1 of $\approx 0.73$. Even when we disable its orthogonal regularization, F1 only rises to 0.78, still significantly lower than a simple LoRA rank 8 (F1 0.834). AdaLoRA's final rank allocation is stable across seeds (Kendall $\tau \approx 0.66$), suggesting that gradient-based importance scores consistently prune directions that are critical for the downstream task.

## 6. Conclusion
This paper evaluated the effect of global LoRA rank under a preregistered equivalence and calibration audit. We find that rank 1 is sufficient for both XLM-R and PhoBERT on UIT-VSFC, while increasing rank induces overconfidence (higher NLL). Critically, we demonstrate that observed "model-specific rank effects" are artifacts of LR tuning and cross-session hardware variance. When controlled via a session-blocked design, the interaction vanishes completely. Finally, a simple global rank 8 outperforms the complex AdaLoRA baseline. Practitioners should prioritize LR tuning and environment control over rank scaling.

**Limitations.** The audit is limited to UIT-VSFC. The fixed data split means test-set noise remains a boundary. We identified a cross-session reproducibility gap for PhoBERT (F1 variance $\approx 0.017$), which we treated as a random effect in our meta-analysis. The T3 analysis used only 4 seeds, limiting its power to detect effects smaller than 0.02.

**Future Work.** Extending this audit to decoder-only LLMs (e.g., Llama) and investigating the root cause of the GPU-specific F1 variance (e.g., CUDA kernels, library versions) would be highly valuable.

## References
- Hu et al. (2022) — LoRA. ICLR.
- Zhang et al. (2023) — AdaLoRA. ICLR.
- Lee et al. (2026) — Learning Rate Matters. arXiv:2602.04998.
- Biderman et al. (2024) — LoRA Learns Less and Forgets Less. TMLR.
- Guo et al. (2017) — On calibration of modern neural networks. ICML.
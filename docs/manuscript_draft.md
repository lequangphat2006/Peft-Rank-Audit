# How Much Does Global LoRA Rank Matter? A Preregistered Equivalence and Calibration Audit on Vietnamese Text Classification

**Le Quang Phat, Dinh Thi Hong Huyen**
*Faculty of Information Technology, Quy Nhon University, Vietnam*

## Abstract

Adaptive rank allocation methods for Low-Rank Adaptation (LoRA) assume that model components have distinct "capacity needs." However, whether increasing the global LoRA rank improves downstream performance under fair hyperparameter tuning remains highly debated. We present a preregistered audit on XLM-R and PhoBERT using the UIT-VSFC dataset, evaluating the global rank from 1 to 32. Using TOST equivalence testing ($\delta = 0.01$), we find that increasing the global rank from 1 to 32 does not improve Macro-F1 for XLM-R. While equivalence holds under seed-level variance (90% CI $[-0.0084, +0.0016]$), a two-tier bootstrap reveals that test-set noise widens the CI to $[-0.013, +0.006]$. Crucially, we expose a hidden confounder: learning rate (LR) selection and cross-session variance. Initial runs suggested a model-specific rank interaction (E1), but a post-hoc session-blocked design yielded a near-zero effect ($\Delta = +0.0014, p_{holm} = 0.70$). A meta-analysis across four iterations shows no evidence different from zero, with a wide CI (HKSJ $[-0.011, +0.015]$) and heterogeneity across iterations (Q $p = 0.022$, $\tau = 0.006$). Furthermore, with default configuration (LR untuned, 3 seeds), a simple global LoRA rank 8 outperforms the AdaLoRA baseline. In our setting, XLM-R's rank effect is LR-dependent, while PhoBERT's prior effect depends on the session (cause unknown). Practitioners should prioritize LR tuning and environment control over complex rank scaling.

## 1. Introduction

Parameter-Efficient Fine-Tuning (PEFT), particularly Low-Rank Adaptation (LoRA), has become the standard for adapting large language models. While vanilla LoRA uses a fixed global rank, recent methods such as AdaLoRA, IGU-LoRA, and StatLoRA dynamically allocate ranks across different modules, assuming that each module possesses a unique and measurable "capacity need." Yet, recent studies suggest that rank-one sufficiency might hold across various settings. This raises a fundamental question: *Does increasing the global LoRA rank actually improve downstream performance?*

A critical confounder in rank comparison is the learning rate (LR). Because LoRA scales its updates by a ratio related to rank (\(\alpha/r\)), higher ranks alter the effective learning dynamics. If the optimal LR shifts with rank, comparing ranks under a fixed LR severely biases the results. Furthermore, we observe that same code and seed can yield a $\approx 0.02$ F1 test deviation for PhoBERT between different sessions; the cause is currently unknown, but it complicates fair rank comparisons. 

In this paper, we conduct a preregistered audit to answer: *How much does global LoRA rank matter for discrimination and calibration when LR is tuned per rank and session effects are controlled?* Our contributions are fourfold:
1. **Preregistered Equivalence Audit:** Using 160 confirmatory runs, we demonstrate that F1 is equivalent from rank 1 to 32 for XLM-R at the seed level. However, a two-tier bootstrap reveals that test-set noise widens the CI to $[-0.013, +0.006]$. Meanwhile, NLL significantly degrades with higher ranks for XLM-R.
2. **The LR Confounder:** The apparent rank-by-epoch interaction reverses sign when LR is fixed (from +0.017 to -0.033). This suggests that the interaction depends on the LR protocol rather than being a stable model property.
3. **Protocol Dependence:** While initial unblocked runs suggested a model-specific rank interaction (E1: $+0.0073, p_{holm} = 0.065$), a post-hoc session-blocked design (T2) yielded a near-zero effect ($\Delta = +0.0014, p_{holm} = 0.70$). Meta-analysis across four iterations shows no evidence different from zero, with wide CI (HKSJ $[-0.011, +0.015]$) and heterogeneity (Q $p = 0.022$; $\tau = 0.006$). 
4. **Adaptive Rank Baseline:** With default configuration (LR 1e-4 taken from LoRA, 3 seeds, no separate tuning), AdaLoRA achieves an F1 of $\approx 0.73$ (0.78 when disabling orthogonal regularization), which is lower than a simple LoRA rank 8 (0.834).

## 2. Related Work

**LoRA and Adaptive Rank Allocation.** Standard LoRA injects trainable rank-decomposition matrices into the model. Building on this, AdaLoRA [Zhang et al., 2023] formulates rank allocation as an SVD-based importance scoring problem. IGU-LoRA [2026] improves this by using integrated gradients, and StatLoRA [Gao et al., 2026] frames rank allocation as statistical hypothesis testing. All these methods assume that a measurable "capacity need" exists.

**Learning Rate and Rank Sufficiency.** Recent work [Lee et al., 2026] demonstrates that when LR is tuned fairly, complex LoRA variants perform equivalently to vanilla LoRA. Furthermore, [Rethinking the Rank Threshold, 2026] shows rank-one sufficiency. Our work differs by using a preregistered equivalence test (TOST) and explicitly isolating the LR and session confounders.

**Calibration in PEFT.** Neural networks are often overconfident [Guo et al., 2017]. While calibration under distribution shift has been studied, the effect of global LoRA rank on NLL and ECE—especially when LR is tuned—remains under-explored.

## 3. Methodology

### 3.1. Dataset and Preprocessing
We conduct our audit on the UIT-VSFC dataset [Nguyen et al., 2018], a Vietnamese student feedback corpus. The dataset is split into 12,780 training, 1,598 calibration, and 1,599 test examples. Near-duplicate analysis (cosine $\ge$ 0.95) identified 3.75% of the test set as near-duplicates; post-hoc checks confirmed that removing them does not change the conclusions.

### 3.2. Global LoRA Rank Scaling
We use `xlm-roberta-base` and `vinai/phobert-base`, injecting LoRA adapters into 72 modules. We vary the global rank \(r \in \{1, 4, 8, 16, 32\}\) with \(\alpha = 2r\). To ensure fairness, LR is tuned strictly on the calibration split using 3 seeds (400-402) and a grid of \(\{1e-4, 2e-4, 5e-4, 1e-3\}\). We also applied a one-step grid extension rule (5e-5 at the lower bound, 2e-3 at the upper). Note: the 5e-5 extension for PhoBERT was run after observing B1X results (post-hoc), and the final LR map remained unchanged (5e-5 did not improve calibration F1).

### 3.3. Preregistered Hypotheses
The confirmatory analysis is structured around H1 (TOST equivalence of F1), H2 (NLL increase), and H3 (rank $\times$ epoch interaction). For the extension (EXT-1), we tested E1 (model $\times$ rank interaction), E2 (PhoBERT equivalence), and E3 (PhoBERT NLL).

### 3.4. Post-hoc Session-Blocked Design and Meta-Analysis
To investigate cross-session variance, we executed a post-hoc session-blocked design (T2) where PhoBERT and XLM-R runs were interleaved in a single session. We also ran T3 (Alternative Data Splits) using 2 additional stratified folds (Fold 2 and Fold 3) with 4 seeds each. A random-effects meta-analysis was applied across the Main, T2, T3-F2, and T3-F3 iterations. Important note: T2_XLM is identical to R1 across 80/80 runs (F1, NLL, and calibration columns match to 6 decimal places), meaning the XLM-R data in B1X and T2 are not independent. 

## 4. Experimental Protocol

### 4.1. Data Splitting and Evaluation Sets
The test split is the sole untouched hold-out. The calibration split is used exclusively for LR selection and temperature scaling. No hyperparameter selection uses the test split.

### 4.2. Configuration, Metrics, and Statistical Testing
**Training:** Batch size 32, max length 128, 3 and 6 epochs, mixed precision (FP16), AdamW optimizer. We report Macro-F1 and unweighted NLL.
**Statistical Models:** We fit OLS regressions using `Metric ~ C(seed) + log2(rank)`. We also run a two-tier bootstrap (resampling both test sentences and seeds).

## 5. Experimental Results

### 5.1. In-Domain Equivalence and Calibration (H1, H2)
**H1 (Equivalence):** For XLM-R, Macro-F1 remains flat across all ranks. TOST confirms equivalence to zero with a 90% CI of \([-0.0084, +0.0016]\) (\(p_{holm} = 0.016\)). A two-tier bootstrap widens the CI to \([-0.013, +0.006]\), which we report as a boundary imposed by test-set noise.
**H2 (Calibration):** For XLM-R, NLL significantly degrades with rank. The total effect is $+0.0049$ (95% CI $[+0.0010, +0.0087]$, \(p_{holm} = 0.014\)). For PhoBERT, the NLL result is inconsistent across sessions (B1X: $-0.0027$, \(p_{holm} = 0.966\); T2: $+0.0113$, \(p_{holm} = 0.0007\)).

### 5.2. The Reversal of Rank $\times$ Epoch (H3)
The preregistered H3 showed a significant interaction (\(p_{holm} = 0.007\)). However, post-hoc diagnostics exposed a confounder. When LR is fixed at 2e-4 across all ranks, the interaction **reverses sign** (from +0.017 to -0.033, $p = 0.0028$). This suggests that the interaction depends on the LR protocol. H3 is **not robust** according to the preregistered rule.

### 5.3. Protocol Dependence and Environment Variance (E1, T2, T3)
**Initial Unblocked Runs:** In the initial main runs (B1X), PhoBERT showed a slightly different F1 slope (F1 0.837 to 0.842), yielding a suggestive model $\times$ rank interaction (E1: $\Delta = +0.0073$, \(p_{holm} = 0.065\)). 
**Session-Blocked Design (T2):** In the post-hoc T2 design, the interaction yielded a near-zero effect (E1: $\Delta = +0.0014$, 95% CI $[-0.006, +0.009]$, \(p_{holm} = 0.70\)). 
**Meta-Analysis (T3):** Across four non-independent iterations (Main, T2, T3-F2, T3-F3), the pooled E1 effect was $+0.0019$. The meta-analysis shows no evidence different from zero, with a wide CI (DerSimonian-Laird with t-distribution: $[-0.0095, +0.0135]$; HKSJ: $[-0.0106, +0.0145]$) and heterogeneity across iterations (Q $p = 0.022$, $\tau = 0.006$). The between-iteration variance exceeds the pooled effect size. Under a fixed-LR protocol, the interaction reversed again (Main branch: $-0.0121$, $p = 0.020$; T2 branch: $-0.0066$, $p = 0.18$).

### 5.4. Mechanistic Analysis: Effective Rank (Delta W)
As nominal rank increased from 1 to 32, the effective rank of the LoRA $\Delta W$ matrices grew nearly linearly from 1.0 to approximately 19.1 (PhoBERT) and 17.1 (XLM-R) at rank 32 in T2. This describes the internal representation dimensions utilized by the model, without making claims about task saturation or noise memorization.

### 5.5. Adaptive Rank Baseline (AdaLoRA)
With a default configuration (LR 1e-4 taken from LoRA, 3 seeds, no separate tuning), AdaLoRA achieves an F1 of $\approx 0.73$ (0.78 when disabling orthogonal regularization). This is lower than a simple LoRA rank 8 (F1 0.834). We make no claims about the underlying mechanism (e.g., pruning critical directions or optimizing on noise) due to limited tuning and seeds.

## 6. Conclusion
This paper evaluated the effect of global LoRA rank under a preregistered equivalence and calibration audit. We find that rank 1 is sufficient for both XLM-R and PhoBERT on UIT-VSFC. NLL increases with rank for XLM-R, while PhoBERT's calibration results are inconsistent across sessions. Critically, observed "model-specific rank effects" are sensitive to LR tuning and cross-session variance. Under a post-hoc session-blocked design, the interaction vanishes. Finally, a simple global rank 8 outperforms the complex AdaLoRA baseline. Practitioners should prioritize LR tuning and environment control over rank scaling.

**Limitations.** The audit is limited to UIT-VSFC. The fixed data split means test-set noise remains a boundary. T2 and T3 are post-hoc analyses; T3 used only 4 seeds, limiting its power to detect effects smaller than 0.02. The XLM-R data in T2 is identical to R1 (not independent). The LR was selected in a different session than the confirmation phase. PhoBERT's LR extension (5e-5) was run after observing B1X results, and the final LR map remained unchanged. AdaLoRA was not separately tuned for LR. The tokenized PhoBERT data is not included in the repository. 

**Future Work.** Extending this audit to decoder-only LLMs (e.g., Llama) and investigating the root cause of the session-specific F1 variance (e.g., CUDA kernels, library versions) would be highly valuable.

## References
- Hu et al. (2022) — LoRA. ICLR.
- Zhang et al. (2023) — AdaLoRA. ICLR.
- Lee et al. (2026) — Learning Rate Matters. arXiv:2602.04998.
- Biderman et al. (2024) — LoRA Learns Less and Forgets Less. TMLR.
- Guo et al. (2017) — On calibration of modern neural networks. ICML.
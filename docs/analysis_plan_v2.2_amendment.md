# Amendment v2.2 (2026-11-15) - deviations and post-hoc analyses

Written after the main confirmatory results were seen. It does not change the registered hypotheses (H1-H3, E1-E3); it records what was added or deviated.

1. **Post-hoc analyses (not registered):** T2 (re-run of the main protocol, seeds 300-315 / 200-215), T3-F2 and T3-F3 (alternative stratified folds, 4 seeds each), the four-iteration random-effects meta-analysis of E1, and the Delta-W effective-rank description (T2 only).
2. **Deviation from the one-step LR extension rule (PhoBERT):** ranks 8, 16 and 32 had best LR 1e-4 at the lower boundary. B1X ran before the extension was applied (`lr_map_best.json`: `extended: false`). 5e-5 was run afterwards, after B1X results were seen, in a slower session (about 340 s per run vs about 219 s).
3. **Same-session control:** 9 runs at 1e-4 (ranks 8/16/32, seeds 400-402, calibration split). Mean calibration macro-F1 1e-4 vs 5e-5: 0.7837 vs 0.7633 (rank 8), 0.7888 vs 0.7834 (rank 16), 0.7967 vs 0.7777 (rank 32); differences +0.0205, +0.0053, +0.0190 vs the registered 0.005 threshold. LR map unchanged. Rank 16 exceeds the threshold by 0.0003 (paired SD 0.0067).
4. **Dependence:** T2_XLM equals R1 on 80/80 runs; B1X and T2 E1 estimates share XLM-R data.
5. **Meta-analysis labels:** DerSimonian-Laird with t(k-1) quantiles and HKSJ are both reported; the earlier z-based interval is not used.
6. **Environment:** no session logs are available; the between-session PhoBERT gap is reported without a causal claim.

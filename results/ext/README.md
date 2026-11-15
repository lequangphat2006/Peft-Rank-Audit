# results/ext - extension (EXT-1) results

Contents were produced by `scripts/run_ext_phobert.sh` (see `docs/ext_runbook.md`) and by the post-hoc runs below.

```text
results/ext/
├── preview/    Dry run of the pipeline on OLD data (not confirmatory evidence)
├── phobert/    tune_*.csv, tuning_decision.json, lr_map_best.json, B1X_total.csv (main runs, seeds 300-315),
│               confirmatory_final_v4.json (E1-E3, main), tune_ctrl_1e4.csv (9 same-session 1e-4 control runs),
│               lr_control_table.json, logits/, deltaw/
├── xlmr/       XLM_total_v2.csv, lr_map_best.json, logits/
├── t2/         T2 post-hoc re-run (not registered): T2_PHO.csv, T2_XLM.csv, confirmatory_official.json, calibration_T2.json, deltaw/
├── t3/         T3 post-hoc alternative folds (4 seeds, low power): T3_*_F2/F3.csv, confirmatory_T3_F2/F3.json
└── meta_analysis.json   E1 meta-analysis across 4 non-independent iterations (python code/ext_fix_all.py)
```

Status notes
- T2 and T3 are post-hoc. T2_XLM equals R1 on 80/80 runs, so B1X and T2 share XLM-R data.
- `tuning_5e5.csv` (results/tuning) was scored on test.csv; use `cal_macro_f1` for LR selection.
- PhoBERT tokenized data is not in the repository.

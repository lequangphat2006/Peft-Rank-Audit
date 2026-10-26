#!/usr/bin/env bash
# run_ext_phobert.sh - chạy phần mở rộng EXT-1 (PhoBERT 5 rank x 16 seed, quy trình LR giống XLM-R) theo từng giai đoạn.
#
# Cách dùng (từ thư mục gốc repo, trên Kaggle hoặc máy có GPU):
#   export DATA_P=/kaggle/input/peft-data/vsfc_sent_phobert      # văn bản đã TÁCH TỪ (text_seg)
#   export DATA_X=/kaggle/input/peft-data/vsfc_sent              # văn bản gốc cho XLM-R
#   export NGPU=2                                                # số GPU (mặc định 2)
#   bash scripts/run_ext_phobert.sh <giai_đoạn>
#
# Giai đoạn theo đúng thứ tự (mỗi giai đoạn có cơ chế resume theo run_id, chạy lại an toàn):
#   preflight_pre   kiểm tra dữ liệu / seed / phiên bản trước khi tốn GPU
#   tune_base       P1  tune LR: 5 rank x 4 LR x 3 seed (400-402) = 60 lượt, chấm trên calibration
#   lr_rule_base    P1  gộp, in ra rank nào cần mở rộng lưới một bước
#   tune_ext        P1  chạy mở rộng: EXT_LOW_RANKS="32 8" EXT_HIGH_RANKS="1" bash ... tune_ext
#   lr_rule_final   P1  gộp toàn bộ tune, chốt LR map v2 (+ bản LR tốt nhì)
#   preflight_main  P1  kiểm tra LR map đủ 5 rank + dry_run đúng 80 lượt
#   final           P1  chạy chính: 5 rank x 16 seed (300-315) x 6 epoch, cell B1X
#   merge_final     P1  gộp và kiểm tra đủ 80 lượt
#   xlmr_maps       P2  xuất LR map phẳng của XLM-R (LR chính và LR tốt nhì) từ tuning_decision.json đã đăng ký
#   robust_fixed    P2  nhánh LR cố định 2e-4 (PhoBERT 5 rank x 8 seed; XLM-R bổ sung seed 204-207)
#   robust_second   P2  nhánh LR tốt nhì (cả hai mô hình, 5 rank x 8 seed)
#   alpha           P2  α = fixed:16, rank {1,8,32}, 8 seed, cả hai mô hình
#   three_epoch     P3  (tùy chọn) PhoBERT 3 epoch để so H3
#   analyze         chạy toàn bộ phân tích và vẽ hình
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
DATA_P="${DATA_P:-}"; DATA_X="${DATA_X:-}"
NGPU="${NGPU:-2}"
MODEL_P="vinai/phobert-base"
OUT_P="$REPO/results/ext/phobert"; OUT_X="$REPO/results/ext/xlmr"
mkdir -p "$OUT_P" "$OUT_X"
RANKS="1 4 8 16 32"
SEEDS_MAIN="300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315"
SEEDS_ROB="300 301 302 303 304 305 306 307"

need() { [ -n "${!1}" ] || { echo "Thiếu biến môi trường $1" >&2; exit 2; }; }

# Chia seed cho GPU theo GIÁ TRỊ seed (seed % NGPU) => seed luôn rơi vào cùng một GPU/CSV ở mọi giai đoạn,
# nên cơ chế resume của run_pilot.py không chạy trùng lượt đã có.
seeds_for() { local g=$1; shift; local o=(); for s in "$@"; do (( s % NGPU == g )) && o+=("$s"); done; echo "${o[@]:-}"; }

# fan_out <outdir> <prefix> <model> <data_dir> <cell> "<seeds>" [tham số run_pilot.py ...]
fan_out() {
  local outdir=$1 prefix=$2 model=$3 data=$4 cell=$5 seeds=$6; shift 6
  local pids=()
  for ((g=0; g<NGPU; g++)); do
    local sg; sg=$(seeds_for "$g" $seeds)
    [ -z "$sg" ] && continue
    CUDA_VISIBLE_DEVICES=$g python code/run_pilot.py --data_dir "$data" --model "$model" \
      --out "$outdir/${prefix}_gpu${g}.csv" --cell "$cell" --seeds $sg --precision fp16 "$@" \
      > "$outdir/${prefix}_gpu${g}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
}

stage="${1:-}"
case "$stage" in
preflight_pre)
  need DATA_P
  python code/ext_preflight.py --lr_map results/v4_phobert/phobert_lr_map.json --ranks 1 4 16 32 \
    --data_phobert "$DATA_P" ${DATA_X:+--data_xlmr "$DATA_X"} --tok_stats \
    --final_seeds 300-315 --tune_seeds 400-402 || true
  echo "(LR map cũ của V4 thiếu rank 8 nên chỉ kiểm tra 4 rank ở bước này; bản v2 sẽ kiểm tra đủ 5 rank.)" ;;

tune_base)
  need DATA_P
  fan_out "$OUT_P" tune_base "$MODEL_P" "$DATA_P" TUNE_PHO "400 401 402" \
    --sweep_global --ranks $RANKS --lr_list 1e-4 2e-4 5e-4 1e-3 --epochs_list 6 --eval_file calibration.csv ;;

lr_rule_base)
  python code/ext_merge.py "$OUT_P/tune_base_gpu*.csv" --out "$OUT_P/tune_base.csv" --cell TUNE_PHO --ranks $RANKS --seeds 400-402
  python code/ext_lr_rule.py "$OUT_P/tune_base.csv" --stage base --data_dir "${DATA_P:-<DATA_P>}" ;;

tune_ext)
  need DATA_P
  [ -n "${EXT_LOW_RANKS:-}" ]  && fan_out "$OUT_P" tune_ext_low  "$MODEL_P" "$DATA_P" TUNE_PHO "400 401 402" \
      --sweep_global --ranks $EXT_LOW_RANKS --lr_list 5e-5 --epochs_list 6 --eval_file calibration.csv
  [ -n "${EXT_HIGH_RANKS:-}" ] && fan_out "$OUT_P" tune_ext_high "$MODEL_P" "$DATA_P" TUNE_PHO "400 401 402" \
      --sweep_global --ranks $EXT_HIGH_RANKS --lr_list 2e-3 --epochs_list 6 --eval_file calibration.csv
  true ;;

lr_rule_final)
  python code/ext_merge.py "$OUT_P/tune_base_gpu*.csv" "$OUT_P/tune_ext_*_gpu*.csv" --out "$OUT_P/tune_all.csv" \
    --cell TUNE_PHO --allow_incomplete
  python code/ext_lr_rule.py "$OUT_P/tune_all.csv" --stage final --out "$OUT_P/tuning_decision.json" \
    --flat_out "$OUT_P/phobert_lr_map_v2.json" --second_out "$OUT_P/phobert_lr_map_v2_second.json" ;;

preflight_main)
  need DATA_P
  python code/ext_preflight.py --lr_map "$OUT_P/phobert_lr_map_v2.json" --data_phobert "$DATA_P" \
    ${DATA_X:+--data_xlmr "$DATA_X"} --final_seeds 300-315 --tune_seeds 400-402 \
    --dry_run_out "$OUT_P/B1X_dryrun.csv" ;;

final)
  need DATA_P
  fan_out "$OUT_P" B1X "$MODEL_P" "$DATA_P" B1X "$SEEDS_MAIN" \
    --sweep_global --ranks $RANKS --epochs_list 6 --lr_map "$OUT_P/phobert_lr_map_v2.json" ;;

merge_final)
  python code/ext_merge.py "$OUT_P/B1X_gpu*.csv" --out "$OUT_P/B1X_merged.csv" --cell B1X \
    --lr_map "$OUT_P/phobert_lr_map_v2.json" --ranks $RANKS --seeds 300-315 --epochs 6 --allow_incomplete ;;

xlmr_maps)
  python code/ext_lr_rule.py results/tuning/tuning_decision.json --stage maps \
    --flat_out "$OUT_X/xlmr_lr_map.json" --second_out "$OUT_X/xlmr_lr_map_second.json" ;;

robust_fixed)
  need DATA_P; need DATA_X
  # PhoBERT: LR cố định 2e-4, 5 rank x 8 seed (các lượt trùng run_id với nhánh chính tự được bỏ qua)
  fan_out "$OUT_P" B1X "$MODEL_P" "$DATA_P" B1X "$SEEDS_ROB" \
    --sweep_global --ranks $RANKS --epochs_list 6 --lr_list 2e-4
  # XLM-R: V1b đã có seed 200-203 (rank 1,4,8,32); rank 16 lấy từ R1 (LR chọn = 2e-4). Bổ sung seed 204-207.
  fan_out "$OUT_X" V1b_ext FacebookAI/xlm-roberta-base "$DATA_X" V1b "204 205 206 207" \
    --sweep_global --ranks 1 4 8 32 --epochs_list 6 --lr_list 2e-4 ;;

robust_second)
  need DATA_P; need DATA_X
  bash "$0" xlmr_maps
  fan_out "$OUT_P" B1X "$MODEL_P" "$DATA_P" B1X "$SEEDS_ROB" \
    --sweep_global --ranks $RANKS --epochs_list 6 --lr_map "$OUT_P/phobert_lr_map_v2_second.json"
  fan_out "$OUT_X" R1B FacebookAI/xlm-roberta-base "$DATA_X" R1B "200 201 202 203 204 205 206 207" \
    --sweep_global --ranks $RANKS --epochs_list 6 --lr_map "$OUT_X/xlmr_lr_map_second.json" ;;

alpha)
  need DATA_P; need DATA_X
  fan_out "$OUT_P" ALPHA_P "$MODEL_P" "$DATA_P" ALPHA_P "$SEEDS_ROB" \
    --sweep_global --ranks 1 8 32 --epochs_list 6 --lr_list 2e-4 --alpha_mode fixed:16
  fan_out "$OUT_X" ALPHA_X FacebookAI/xlm-roberta-base "$DATA_X" ALPHA_X "200 201 202 203 204 205 206 207" \
    --sweep_global --ranks 1 8 32 --epochs_list 6 --lr_list 2e-4 --alpha_mode fixed:16 ;;

three_epoch)
  need DATA_P
  fan_out "$OUT_P" B1X "$MODEL_P" "$DATA_P" B1X "$SEEDS_MAIN" \
    --sweep_global --ranks $RANKS --epochs_list 3 --lr_map "$OUT_P/phobert_lr_map_v2.json" ;;

analyze)
  XCSV=(results/global/global_merged.csv results/posthoc/V1b_merged.csv)
  for f in "$OUT_X"/V1b_ext_gpu*.csv "$OUT_X"/R1B_gpu*.csv; do [ -f "$f" ] && XCSV+=("$f"); done
  python code/ext_analyze_phobert.py "$OUT_P/B1X_merged.csv" --lr_map "$OUT_P/phobert_lr_map_v2.json" \
    --decision "$OUT_P/tuning_decision.json" --legacy_lr_map results/v4_phobert/phobert_lr_map.json \
    --out "$OUT_P/analysis_phobert.json"
  python code/ext_confirmatory.py --csv_p "$OUT_P/B1X_merged.csv" --lr_map_p "$OUT_P/phobert_lr_map_v2.json" \
    --decision_p "$OUT_P/tuning_decision.json" --csv_x "${XCSV[@]}" --lr_map_x results/tuning/tuning_decision.json \
    --decision_x results/tuning/tuning_decision.json --logits_p "$OUT_P/logits" --logits_x results/global/logits \
    --out "$OUT_P/confirmatory.json"
  python code/ext_calibration.py --csv_p "$OUT_P/B1X_merged.csv" --lr_map_p "$OUT_P/phobert_lr_map_v2.json" \
    --logits_p "$OUT_P/logits" --csv_x results/global/global_merged.csv --lr_map_x results/tuning/tuning_decision.json \
    --logits_x results/global/logits --out "$OUT_P/calibration.json"
  python code/ext_deltaw.py --csv_p "$OUT_P/B1X_merged.csv" --lr_map_p "$OUT_P/phobert_lr_map_v2.json" \
    --deltaw_p "$OUT_P/deltaw" --csv_x results/global/global_merged.csv --lr_map_x results/tuning/tuning_decision.json \
    --deltaw_x results/global/deltaw --out "$OUT_P/deltaw_summary.csv" || true
  python code/ext_alpha.py --csv_p "$OUT_P/B1X_merged.csv" "$OUT_P"/ALPHA_P_gpu*.csv \
    --csv_x results/posthoc/V1b_merged.csv "$OUT_X"/V1b_ext_gpu*.csv "$OUT_X"/ALPHA_X_gpu*.csv \
    --out "$OUT_P/alpha.json" || true
  python code/ext_make_figures.py --csv_p "$OUT_P/B1X_merged.csv" --lr_map_p "$OUT_P/phobert_lr_map_v2.json" \
    --csv_x results/global/global_merged.csv --lr_map_x results/tuning/tuning_decision.json \
    --tune_p "$OUT_P/tune_all.csv" --decision_p "$OUT_P/tuning_decision.json" \
    --confirmatory "$OUT_P/confirmatory.json" --outdir figures --update_manifest ;;

*)
  sed -n '2,25p' "$0"; exit 1 ;;
esac

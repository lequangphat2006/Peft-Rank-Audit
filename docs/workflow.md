# Workflow
> **Mọi session Kaggle mới:** đọc và làm theo `docs/SETUP_KAGGLE.md` trước khi chạy bất kỳ cell nào.

## Cycle

### 1. Local — prepare
- git pull
- pytest
- prepare_vsfc.py
- git push

### 2. Kaggle — train
- Update Dataset peft-code
- Mount peft-code, peft-data
- Chạy notebook
- Save Version

### 3. Local — pull
- pull_kaggle.ps1
- merge_results.py
- analyze_pilot.py

### 4. Local — update + push
- update_readme.py
- push_cycle.ps1

## Resume

run_pilot.py đọc file --out nếu có, bỏ qua lượt đã chạy.

## Khi Kaggle bị ngắt

1. Save Version lại
2. Upload output cũ vào Dataset peft-results
3. Mount vào notebook
4. Chạy lại, script tự resume

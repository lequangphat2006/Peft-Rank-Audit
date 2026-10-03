# Pull results từ Kaggle về local
# Usage: bash scripts/pull_kaggle.sh <session_number>
$SESSION = $args[0]
if (-not $SESSION) {
    Write-Host "Usage: pull_kaggle.ps1 <session_number>"
    exit 1
}
New-Item -ItemType Directory -Force -Path "results/pilot" | Out-Null
kaggle kernels output lequangphat2006/peft-run-pilot -p results/pilot/ --file-pattern "results_C1.csv"
if (Test-Path "results/pilot/results_C1.csv") {
    Move-Item "results/pilot/results_C1.csv" "results/pilot/results_C1_session${SESSION}.csv" -Force
    Write-Host "Pulled session $SESSION"
    Get-ChildItem results/pilot/
} else {
    Write-Host "Không tìm thấy results_C1.csv"
}

# Push cycle lên GitHub
# Usage: .\scripts\push_cycle.ps1 <session> <cycle>
$SESSION = $args[0]
$CYCLE = $args[1]
if (-not $SESSION -or -not $CYCLE) {
    Write-Host "Usage: push_cycle.ps1 <session> <cycle>"
    exit 1
}
git add README.md CHANGELOG.md code/ notebooks/ results/ docs/ scripts/
git commit -m "Cycle $CYCLE, Session $SESSION"
git push origin main
Write-Host "Pushed cycle $CYCLE"

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

Write-Host "[run_ui] Repository root: $repoRoot"

foreach ($required in @("main.py", "requirements.txt", ".env.example")) {
    if (-not (Test-Path $required)) {
        throw "[run_ui] Missing required file: $required. Please run from the repository checkout."
    }
}

Write-Host "[run_ui] Installing dependencies..."
python -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Write-Host "[run_ui] .env not found, creating from .env.example"
    Copy-Item ".env.example" ".env"
}

Write-Host "[run_ui] Starting Streamlit UI..."
python -m streamlit run main.py

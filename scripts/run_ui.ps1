$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

Write-Host "[run_ui] Repository root: $repoRoot"

foreach ($required in @("main.py", "requirements.txt", ".env.example")) {
    if (-not (Test-Path $required)) {
        throw "[run_ui] Missing required file: $required. Please run this script from the repository root directory."
    }
}

if ($env:RUN_UI_SKIP_INSTALL -eq "1") {
    Write-Host "[run_ui] Skipping dependency install because RUN_UI_SKIP_INSTALL=1"
} else {
    Write-Host "[run_ui] Installing dependencies..."
    python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        throw "[run_ui] Dependency installation failed. Verify Python/pip installation and network connectivity."
    }
}

if (-not (Test-Path ".env")) {
    Write-Host "[run_ui] .env not found, creating from .env.example"
    try {
        Copy-Item ".env.example" ".env" -ErrorAction Stop
    } catch {
        throw "[run_ui] Failed to create .env from .env.example: $($_.Exception.Message)"
    }
}

Write-Host "[run_ui] Starting Streamlit UI..."
python -m streamlit run main.py

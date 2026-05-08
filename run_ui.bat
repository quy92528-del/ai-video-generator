@echo off
setlocal

set "REPO_DIR=%~dp0"
cd /d "%REPO_DIR%"

echo [run_ui] Repository root: %CD%

if not exist "main.py" (
  echo [run_ui] Missing required file: main.py
  exit /b 1
)
if not exist "requirements.txt" (
  echo [run_ui] Missing required file: requirements.txt
  exit /b 1
)
if not exist ".env.example" (
  echo [run_ui] Missing required file: .env.example
  exit /b 1
)

if /I "%RUN_UI_SKIP_INSTALL%"=="1" (
  echo [run_ui] Skipping dependency install because RUN_UI_SKIP_INSTALL=1
) else (
  echo [run_ui] Installing dependencies...
  python -m pip install -r requirements.txt || exit /b 1
)

if not exist ".env" (
  echo [run_ui] .env not found, creating from .env.example
  copy ".env.example" ".env" >nul || (
    echo [run_ui] Failed to create .env from .env.example
    exit /b 1
  )
)

echo [run_ui] Starting Streamlit UI...
python -m streamlit run main.py

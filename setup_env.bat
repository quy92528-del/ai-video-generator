@echo off
echo Installing AI Video Generator dependencies...
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
python -m pip install --no-cache-dir -r requirements.txt
if errorlevel 1 exit /b 1
echo Installation complete!
pause

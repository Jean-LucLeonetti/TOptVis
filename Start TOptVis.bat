@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"

if not exist ".venv\Scripts\python.exe" (
    echo The local Python environment is missing. Run scripts\setup_windows.ps1 first.
    pause
    exit /b 1
)

echo Starting TOptVis. Keep this window open while using the app.
".venv\Scripts\python.exe" -m ml_opt.web --open-browser
if errorlevel 1 (
    echo.
    echo TOptVis could not start. See the error above.
    pause
)

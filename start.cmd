@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    call setup.cmd
    if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -c "import simpy, streamlit, pandas, plotly" >nul 2>&1
if errorlevel 1 (
    call setup.cmd
    if errorlevel 1 exit /b 1
)
echo Starting Building Flow. Keep this window open while using the app.
echo Close this window or press Ctrl+C to stop the local server.
".venv\Scripts\python.exe" eos.py
if errorlevel 1 pause

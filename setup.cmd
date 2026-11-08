@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv ".venv"
    if errorlevel 1 (
        echo Please install Python 3.12 and retry.
        pause
        exit /b 1
    )
)
".venv\Scripts\python.exe" -m pip --version >nul 2>&1
if errorlevel 1 ".venv\Scripts\python.exe" -m ensurepip --upgrade
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.lock.txt
if errorlevel 1 (
    echo Dependency setup failed. Check the network connection and retry setup.cmd.
    pause
    exit /b 1
)
echo Dependencies are ready.

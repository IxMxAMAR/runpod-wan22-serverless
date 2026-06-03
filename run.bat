@echo off
REM Launch the WAN 2.2 Serverless GUI. Double-click or run from cmd.
cd /d "%~dp0"
python gui.py
if errorlevel 1 (
    echo.
    echo GUI exited with error. Press any key to close.
    pause >nul
)

@echo off
title FLOW System V3.0
cd /d "%~dp0"

:loop
cls
echo =======================================
echo   FLOW System V3.0 - Starting...
echo   Press Ctrl+C to stop or restart
echo =======================================
echo.
streamlit run main.py

echo.
echo FLOW System stopped.
echo Press any key to restart, or close this window to exit.
pause >nul
goto loop

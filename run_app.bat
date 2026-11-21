@echo off
REM Starts the app at http://localhost:8501 and opens it in the default browser.
REM Runs headless (no first-run email prompt); the browser is opened by this script.
cd /d "%~dp0"
if not exist "%~dp0tmp" mkdir "%~dp0tmp"
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYTHONUTF8=1
set TMP=%~dp0tmp
set TEMP=%~dp0tmp
start "" /b cmd /c "%SystemRoot%\System32\PING.EXE -n 7 127.0.0.1 >nul & start "" http://localhost:8501"
"%~dp0.venv\Scripts\streamlit.exe" run app.py --server.headless true

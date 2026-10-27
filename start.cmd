@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" run.py %*
) else if exist "..\.venv-q1\Scripts\python.exe" (
  "..\.venv-q1\Scripts\python.exe" run.py %*
) else (
  python run.py %*
)
if errorlevel 1 pause
endlocal

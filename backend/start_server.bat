@echo off
title AI-Generated Text Detection API Server (FastAPI)
echo ======================================================================
echo   KHOI DONG FASTAPI BACKEND SERVER - VERITEXT AI
echo ======================================================================
echo.

cd /d "%~dp0"

if exist "..\venv\Scripts\python.exe" (
    set "PYTHON_EXEC=..\venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXEC=venv\Scripts\python.exe"
) else (
    echo [!] LOI: Khong tim thay moi truong ao venv!
    pause
    exit /b 1
)

echo [*] Dang khoi dong server va nap mo hinh, vui long cho...
%PYTHON_EXEC% run_server.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Co loi xay ra khi chay server.
    pause
)

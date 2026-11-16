@echo off
title AI-Generated Text Detection API Server (FastAPI)
echo ======================================================================
echo   KHOI DONG FASTAPI BACKEND SERVER - DO AN CHUYEN NGANH
echo   Mo hinh Ensemble PLMs: BERT + RoBERTa + Longformer
echo ======================================================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [!] LOI: Khong tim thay moi truong ao venv!
    pause
    exit /b 1
)

echo [*] Dang khoi dong server va nap mo hinh, vui long cho...
venv\Scripts\python.exe run_server.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Co loi xay ra khi chay server.
    pause
)

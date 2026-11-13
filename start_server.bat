@echo off
chcp 65001 > nul
title AI-Generated Text Detection API Server (FastAPI)
echo ======================================================================
echo   KHỞI ĐỘNG FASTAPI BACKEND SERVER - ĐỒ ÁN CHUYÊN NGÀNH
echo   Mô hình Ensemble PLMs: BERT + RoBERTa + Longformer
echo ======================================================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [!] LỖI: Không tìm thấy môi trường ảo venv! Vui lòng cài đặt trước.
    pause
    exit /b 1
)

echo [*] Đang kích hoạt môi trường ảo Python...
echo [*] Khởi động Uvicorn server tại : http://localhost:8000
echo [*] Tài liệu Swagger UI tương tác : http://localhost:8000/docs
echo [*] Nhấn Ctrl+C để dừng server.
echo.

start http://localhost:8000/docs
venv\Scripts\python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000

pause


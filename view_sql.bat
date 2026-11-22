@echo off
title VeriText AI - SQL Server Database Viewer
echo ======================================================================
echo   XEM DU LIEU TRONG MICROSOFT SQL SERVER 2022 (VERITEXTAI)
echo ======================================================================
echo.

cd /d "%~dp0"
venv\Scripts\python.exe backend\view_db.py

echo.
echo ======================================================================
pause

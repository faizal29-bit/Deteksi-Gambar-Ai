@echo off
chcp 65001 >nul
title AI Image Detector

echo.
echo =====================================================
echo   AI Image Detector  -  Starting Server...
echo =====================================================
echo.

REM ── Activate virtual environment (jika ada) ──────────────────────────
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Mengaktifkan virtual environment (venv)...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Mengaktifkan virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else (
    echo [WARN] Virtual environment tidak ditemukan, menggunakan Python global.
)

echo.

REM ── Cek Python ────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python tidak ditemukan. Pastikan Python sudah terinstal.
    pause
    exit /b 1
)

REM ── Cek model files ───────────────────────────────────────────────────
if not exist "model\model.safetensors" (
    echo [ERROR] File model\model.safetensors tidak ditemukan!
    echo         Pastikan folder model\ berisi file model yang sudah dilatih.
    pause
    exit /b 1
)
if not exist "model\config.json" (
    echo [ERROR] File model\config.json tidak ditemukan!
    pause
    exit /b 1
)

echo [OK] File model ditemukan.
echo.
echo [INFO] Server berjalan di: http://localhost:5000
echo [INFO] Tekan Ctrl+C untuk menghentikan server.
echo.
echo =====================================================
echo.

REM ── Jalankan server ───────────────────────────────────────────────────
python app.py

echo.
echo [INFO] Server dihentikan.
pause

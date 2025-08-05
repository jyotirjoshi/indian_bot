@echo off
echo ========================================
echo    ENHANCED TRADING BOT LAUNCHER
echo ========================================
echo.

cd /d "%~dp0"

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Installing required packages...
pip install -r requirements_enhanced.txt >nul 2>&1

echo.

echo Starting Enhanced Trading Bot...
echo.
echo Options:
echo 1. Interactive Mode (recommended)
echo 2. Auto Mode (starts bot and dashboard automatically)
echo 3. Status Only
echo.

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    python start_enhanced_bot.py
) else if "%choice%"=="2" (
    python start_enhanced_bot.py --auto
) else if "%choice%"=="3" (
    python start_enhanced_bot.py --status
) else (
    echo Invalid choice. Starting in interactive mode...
    python start_enhanced_bot.py
)

echo.
echo Bot stopped. Press any key to exit...
pause >nul
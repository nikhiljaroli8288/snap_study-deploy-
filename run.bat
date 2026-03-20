@echo off
echo ====================================
echo    StudySnap AI - Starting Server
echo ====================================
echo.

cd /d "%~dp0"

echo Activating virtual environment...
call venv\Scripts\activate

echo Starting Flask server...
echo.
echo Server will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python app.py

pause

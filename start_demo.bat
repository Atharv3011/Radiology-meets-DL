@echo off
echo.
echo ============================================================
echo 🩻 FractureDetect AI - Professional Final Year Project
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found. Running setup first...
    echo.
    python scripts\setup_project.py
    echo.
)

echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo 🚀 Starting FractureDetect AI servers...
echo.

REM Start backend server in background
echo 📡 Starting backend API server...
start "FractureDetect API" /min cmd /c "call venv\Scripts\activate.bat && python backend\demo_app.py"

REM Wait a moment for backend to start
echo ⏳ Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

REM Start frontend server
echo 🌐 Starting frontend server...
cd frontend
start "FractureDetect Frontend" /min cmd /c "python -m http.server 8080"

REM Wait a moment for frontend to start
echo ⏳ Waiting for frontend to initialize...
timeout /t 3 /nobreak >nul

echo.
echo ✅ FractureDetect AI is now running!
echo ============================================================
echo 🌐 Frontend URL:     http://localhost:8080/enhanced_index.html
echo 📡 Backend API:      http://localhost:5000
echo 🧪 System Test:      http://localhost:8080/test_system.html
echo 📊 Health Check:     http://localhost:5000/health
echo ============================================================
echo.
echo 📋 Instructions:
echo   1. Open the Frontend URL in your web browser
echo   2. Upload an X-ray image (JPG, PNG, etc.)
echo   3. Click "Analyze for Fractures" to see AI prediction
echo   4. View the detailed results and explanations
echo.
echo ⚠️  NOTE: This is a DEMO version with simulated predictions
echo    Real AI models require PyTorch installation (see docs)
echo.
echo 🛑 To stop the servers, close the terminal windows
echo    or press Ctrl+C in each server window
echo.

REM Open browser automatically
echo 🌐 Opening browser automatically...
timeout /t 2 /nobreak >nul
start http://localhost:8080/enhanced_index.html

echo.
echo 🎉 Ready to demonstrate your AI project!
echo.
pause
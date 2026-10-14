@echo off
setlocal
cd /d "%~dp0"
set "ROOT=%~dp0"

set "PY=python"
if exist "%ROOT%venv\Scripts\python.exe" set "PY=%ROOT%venv\Scripts\python.exe"
if exist "%ROOT%.venv\Scripts\python.exe" set "PY=%ROOT%.venv\Scripts\python.exe"
if exist "%ROOT%backend\venv\Scripts\python.exe" set "PY=%ROOT%backend\venv\Scripts\python.exe"
if exist "%ROOT%backend\.venv\Scripts\python.exe" set "PY=%ROOT%backend\.venv\Scripts\python.exe"

set "FE="
if exist "%ROOT%frontend\package.json" set "FE=%ROOT%frontend"
if not defined FE if exist "%ROOT%package.json" set "FE=%ROOT%"
if not defined FE (
  echo ERROR: no package.json found in project root or frontend folder.
  pause & exit /b 1
)
if not exist "%FE%\node_modules" (
  echo ERROR: frontend dependencies not installed. Run "npm install" in %FE% once.
  pause & exit /b 1
)

netstat -ano | findstr ":8000 " | findstr LISTENING >nul && (
  echo Port 8000 is already in use. Run stop_ai_archaeologist.bat first.
  pause & exit /b 1
)
netstat -ano | findstr ":5173 " | findstr LISTENING >nul && (
  echo Port 5173 is already in use. Run stop_ai_archaeologist.bat first.
  pause & exit /b 1
)

echo Using Python: %PY%
echo Frontend dir: %FE%
start "AI Archaeologist - Backend" /D "%ROOT%backend" cmd /k ""%PY%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"
start "AI Archaeologist - Frontend" /D "%FE%" cmd /k "npm run dev"

echo.
echo Backend docs : http://127.0.0.1:8000/docs
echo Frontend     : http://localhost:5173  (check the Frontend window if Vite shows another port)
timeout /t 8 >nul
start http://localhost:5173

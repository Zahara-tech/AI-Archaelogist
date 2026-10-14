@echo off
powershell -NoProfile -Command "foreach($p in 8000,5173){Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }}"
echo Stopped servers on ports 8000 and 5173.
pause

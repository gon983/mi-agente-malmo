@echo off
REM Script para lanzar Minecraft Malmo y el agente automaticamente

echo ============================================================
echo    AGENTE DE MALMO - Launcher
echo ============================================================
echo.

REM Lanzar Minecraft en una nueva ventana
echo [1/2] Iniciando Minecraft con MalmoMod en puerto 9000...
start "Minecraft Malmo" cmd /k "cd /d c:\Users\gonza\malmo\Minecraft && launchClient.bat -port 9000 -env"

REM Esperar a que Minecraft inicie (60 segundos)
echo [2/2] Esperando 60 segundos para que Minecraft inicie...
echo       (Cierra esta ventana si prefieres esperar manualmente)
timeout /t 60 /nobreak

echo.
echo Ejecutando el agente...
cd /d C:\Users\gonza\agentedemalmo
python run_agent.py --episodes 3 --max-steps 30

echo.
pause
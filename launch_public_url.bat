@echo off
title Distracted Driver Detection System - Live Public URL Launcher
echo =====================================================================
echo  VISION-BASED DISTRACTED DRIVER DETECTION SYSTEM
echo  Launching Local Server and Generating Live Public HTTPS Link...
echo =====================================================================

cd /d "%~dp0"
start "Flask Server" cmd /k ".\venv\Scripts\python.exe app.py"

echo Waiting for Flask server to initialize...
timeout /t 3 /nobreak >nul

echo Starting Public Tunnel via LocalTunnel...
echo Your public link will appear below! Click or copy it into your browser.
echo =====================================================================
npx localtunnel --port 5000
pause

@echo off
title Parkinson Prediction App

echo Starting Streamlit app...
start "" cmd /c "streamlit run web\app.py"

echo Waiting for server to start...
timeout /t 3 /nobreak >nul

echo Opening Chrome...
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" http://localhost:8501

echo App is running. Close this window to stop.
pause

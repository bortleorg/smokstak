@echo off
setlocal
cd /d "%~dp0"
echo Starting smokstak. Keep this window open while stacking.
echo Close the window or press Ctrl+C to stop the local web app.
"%~dp0smokstak.exe" gui
if errorlevel 1 pause

@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0upload-benchmark-results.ps1" %*
exit /b %errorlevel%

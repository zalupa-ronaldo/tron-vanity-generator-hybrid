@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0test-opencl.ps1" %*
set "test_exit=%errorlevel%"
pause
exit /b %test_exit%

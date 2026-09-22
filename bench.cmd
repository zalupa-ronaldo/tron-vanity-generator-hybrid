@echo off
rem One-command, no-wallet RX 9070 XT benchmark. Each GPU variant has its own timeout.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0test-opencl.ps1" -All -TimeoutSeconds 120 %*
set "bench_exit=%errorlevel%"
pause
exit /b %bench_exit%

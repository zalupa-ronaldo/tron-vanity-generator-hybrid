@echo off
rem Bounded, no-wallet OpenCL/Vulkan A/B benchmark from the opt-in artifact.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bench-vulkan.ps1" %*
set "bench_exit=%errorlevel%"
pause
exit /b %bench_exit%

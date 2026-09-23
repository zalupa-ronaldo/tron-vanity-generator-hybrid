@echo off
rem One-command, no-wallet RX 9070 XT benchmark. Every child has its own timeout.
rem Keep running the second suite if the first fails: each suite reports its
rem own correctness gate and the failure may be backend-specific.
call powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0test-opencl.ps1" -All -TimeoutSeconds 120
set "opencl_exit=%errorlevel%"
call powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bench-vulkan.ps1" -TimeoutSeconds 120
set "vulkan_exit=%errorlevel%"
echo.
echo OpenCL suite exit: %opencl_exit%  Vulkan suite exit: %vulkan_exit%
echo Send the summary.txt and benchmark.csv from both report folders only.
set "bench_exit=0"
if not "%opencl_exit%"=="0" set "bench_exit=1"
if not "%vulkan_exit%"=="0" set "bench_exit=1"
if not defined TRON_BENCH_NO_PAUSE pause
exit /b %bench_exit%

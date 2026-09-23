@echo off
rem Bounded, no-wallet OpenCL/Vulkan A/B benchmark from the Windows ZIP.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bench-vulkan.ps1" %*
set "bench_exit=%errorlevel%"
if defined TRON_BENCH_UPLOAD_TOKEN (
  call powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\upload-benchmark-results.ps1"
  if not "%errorlevel%"=="0" set "bench_exit=1"
)
pause
exit /b %bench_exit%

# Build with an installed Visual Studio (or VS Build Tools) + its bundled CMake/Ninja.
# Plain PowerShell, no "Developer Command Prompt" needed:
#   powershell -ExecutionPolicy Bypass -File build.ps1
#
# Requires: Visual Studio 2022/2026 or Build Tools with the C++ workload, git, GitHub access.
# The resulting exe statically links the runtime and can be copied to another x64 Windows PC.
param(
    [string]$Config = "Release",
    [switch]$EnableCuda,
    [switch]$EnableVulkan,
    [switch]$SkipVulkanRuntimeTests
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path "$root\third_party\secp256k1\CMakeLists.txt")) {
    Write-Host "Fetching libsecp256k1 submodule ..."
    if (Test-Path "$root\.git") {
        git -C $root submodule update --init --recursive
    } else {
        git clone --depth 1 https://github.com/bitcoin-core/secp256k1.git "$root\third_party\secp256k1"
    }
}

# --- locate Visual Studio ---
$installerDir = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer"
$vswhere = Join-Path $installerDir "vswhere.exe"
$vsPath = $null
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
if (-not $vsPath) {
    foreach ($p in @("D:\devtools\VS2026BuildTools",
                     "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools",
                     "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community",
                     "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional")) {
        if (Test-Path "$p\VC\Auxiliary\Build\vcvars64.bat") { $vsPath = $p; break }
    }
}
if (-not $vsPath) { throw "Visual Studio / Build Tools not found (need the C++ workload)" }
Write-Host "VS: $vsPath"

# vcvars calls vswhere internally; put the Installer dir on PATH so a fresh env resolves it
$env:PATH = "$installerDir;$env:PATH"

$cmake = Join-Path $vsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if (-not (Test-Path $cmake)) { $cmake = "cmake" }
$ninja = Join-Path $vsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"

$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}
if (-not $env:INCLUDE) { throw "MSVC environment not loaded (INCLUDE is empty)" }

if (Test-Path $ninja) {
    $gen = @("-G", "Ninja", "-DCMAKE_MAKE_PROGRAM=$ninja")
} else {
    $gen = @("-G", "NMake Makefiles")
}

$cudaOption = if ($EnableCuda) { "ON" } else { "OFF" }
$vulkanOption = if ($EnableVulkan) { "ON" } else { "OFF" }
if ($EnableVulkan -and -not $env:VULKAN_SDK) {
    throw "-EnableVulkan requires the LunarG Vulkan SDK and VULKAN_SDK environment variable"
}
& $cmake -B build $gen "-DCMAKE_BUILD_TYPE=$Config" "-DTRON_ENABLE_CUDA=$cudaOption" "-DTRON_ENABLE_VULKAN=$vulkanOption"
if ($LASTEXITCODE) { throw "cmake configure failed" }
& $cmake --build build --config $Config
if ($LASTEXITCODE) { throw "build failed" }
if ($SkipVulkanRuntimeTests) {
    $cmakeDir = if (Test-Path $cmake) { Split-Path $cmake } else { $null }
    $ctest = if ($cmakeDir -and (Test-Path (Join-Path $cmakeDir "ctest.exe"))) {
        Join-Path $cmakeDir "ctest.exe"
    } else { "ctest" }
    & $ctest --test-dir build -C $Config -E '^vulkan_' --output-on-failure
} else {
    & $cmake --build build --config $Config --target test
}
if ($LASTEXITCODE) { throw "kernel correctness tests failed" }

Write-Host ""
Write-Host "Done: $root\build\tron_vanity_generator.exe"
& "$root\build\tron_vanity_generator.exe" --selftest
if ($LASTEXITCODE) { throw "selftest failed" }
& "$root\build\tron_vanity_generator.exe" --hashtest
if ($LASTEXITCODE) { throw "hashtest failed" }
& "$root\build\tron_vanity_generator.exe" --matchtest
if ($LASTEXITCODE) { throw "matchtest failed" }
& "$root\build\tron_vanity_generator.exe" --list
if ($LASTEXITCODE) { throw "device listing failed" }

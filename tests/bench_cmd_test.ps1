# Exercise the packaged one-click orchestrator without a GPU or wallet files.
$ErrorActionPreference = "Stop"
$root = Join-Path ([IO.Path]::GetTempPath()) ("tron-bench-cmd-test-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $root | Out-Null
$priorNoPause = $env:TRON_BENCH_NO_PAUSE
$priorFixture = $env:TRON_BENCH_FIXTURE
try {
    foreach ($case in @(
        @{ Name = "success"; Exit = 0 },
        @{ Name = "fail-opencl"; Exit = 1 },
        @{ Name = "fail-vulkan"; Exit = 1 }
    )) {
        $dir = Join-Path $root ("test " + $case.Name)
        New-Item -ItemType Directory -Path $dir | Out-Null
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot "../bench.cmd") -Destination $dir
        Set-Content -LiteralPath (Join-Path $dir "powershell.cmd") -Encoding ASCII -Value @'
@echo off
echo %*>>calls.txt
echo %*|findstr /C:"test-opencl.ps1" >nul
if not errorlevel 1 if "%TRON_BENCH_FIXTURE%"=="fail-opencl" exit /b 7
echo %*|findstr /C:"bench-vulkan.ps1" >nul
if not errorlevel 1 if "%TRON_BENCH_FIXTURE%"=="fail-vulkan" exit /b 8
exit /b 0
'@
        $env:TRON_BENCH_NO_PAUSE = "1"
        $env:TRON_BENCH_FIXTURE = $case.Name
        Push-Location $dir
        try {
            $output = & cmd.exe /d /c bench.cmd 2>&1
            $exitCode = $LASTEXITCODE
            $calls = @(Get-Content -LiteralPath (Join-Path $dir "calls.txt"))
        } finally { Pop-Location }
        if ($exitCode -ne $case.Exit) {
            throw "$($case.Name) exit mismatch: $exitCode`n$($output -join "`n")"
        }
        if ($calls.Count -ne 2 -or $calls[0] -notmatch 'test-opencl\.ps1' -or
            $calls[1] -notmatch 'bench-vulkan\.ps1') {
            throw "$($case.Name) did not run both suites in order: $($calls -join '; ')"
        }
        if ($output -notmatch 'Send the summary.txt and benchmark.csv from both report folders') {
            throw "$($case.Name) did not explain which reports to share"
        }
    }
    Write-Host "One-click benchmark orchestration PASS (success and both failure paths)"
} finally {
    $env:TRON_BENCH_NO_PAUSE = $priorNoPause
    $env:TRON_BENCH_FIXTURE = $priorFixture
    if (Test-Path -LiteralPath $root) { [IO.Directory]::Delete($root, $true) }
}
exit 0

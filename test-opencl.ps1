# Runs only self-tests/profiles. Never generates a wallet file or prints keys.
param(
    [ValidateSet("single", "pair")][string]$Inverse = "pair",
    [ValidateSet("compact", "default")][string]$Compiler = "compact",
    [ValidateRange(30, 600)][int]$TimeoutSeconds = 120
)
$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "tron_vanity_generator.exe"
if (-not (Test-Path $exe)) { throw "Extract the release ZIP before running this script." }

function Invoke-BoundedTest([string[]]$TestArguments) {
    Write-Host ("Running: " + ($TestArguments -join " "))
    $process = Start-Process -FilePath $exe -ArgumentList $TestArguments -WorkingDirectory $PSScriptRoot -NoNewWindow -PassThru
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    while (-not $process.WaitForExit(5000)) {
        Write-Host ("Still running: {0:N0} s (limit {1} s)" -f $timer.Elapsed.TotalSeconds, $TimeoutSeconds)
        if ($timer.Elapsed.TotalSeconds -ge $TimeoutSeconds) {
            # Stop only the child created above, never another worker or driver.
            $process.Kill()
            $process.WaitForExit()
            throw "OpenCL test timed out. The test process was stopped; no wallet files were created."
        }
    }
    if ($process.ExitCode -ne 0) { throw "OpenCL test failed with exit code $($process.ExitCode)." }
}

$common = @("--backend", "opencl", "--opencl-compiler", $Compiler, "--opencl-inverse", $Inverse)
Invoke-BoundedTest ($common + @("--gpu-resident", "--gputest"))
Invoke-BoundedTest ($common + @("--opencl-profile", "--words", "words.txt", "--gpu-buffer-mb", "8", "--bench-seconds", "5"))
Write-Host "OpenCL self-test and profile completed. Copy the timing output, not any wallet files."

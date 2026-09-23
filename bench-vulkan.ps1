# Opt-in RX 9070 XT comparison. Only no-wallet self-test/profile commands run.
param(
    [ValidateRange(30, 600)][int]$TimeoutSeconds = 120,
    [ValidateRange(1, 60)][int]$Seconds = 5,
    [ValidateSet(32768, 65536, 131072, 262144)][int]$VulkanBatchKeys = 131072
)
$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "tron_vanity_generator.exe"
$words = Join-Path $PSScriptRoot "words.txt"
if (-not (Test-Path -LiteralPath $exe)) { throw "Extract the complete Windows ZIP next to this script first." }
if (-not (Test-Path -LiteralPath $words)) { throw "Put the same words.txt used by OpenCL next to the exe." }

$logDir = Join-Path $PSScriptRoot ("vulkan-benchmark-" + (Get-Date -Format "yyyyMMdd-HHmmss") + "-" +
                                  [guid]::NewGuid().ToString("N").Substring(0, 6))
New-Item -ItemType Directory -Path $logDir | Out-Null
$summary = Join-Path $logDir "summary.txt"
$results = [System.Collections.Generic.List[object]]::new()
$stopAfterTimeout = $false

function Write-Report([string]$Text) {
    Write-Host $Text
    Add-Content -LiteralPath $summary -Value $Text -Encoding UTF8
}

function Invoke-Bounded([string]$Name, [string[]]$Arguments) {
    Write-Report ("`n[$Name] " + ($Arguments -join " "))
    $start = [System.Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $exe
    # All arguments are fixed CLI tokens or validated integers without spaces.
    # The dictionary path is relative to WorkingDirectory, so a spaced ZIP
    # extraction directory does not need Windows command-line escaping.
    $start.Arguments = $Arguments -join " "
    $start.WorkingDirectory = $PSScriptRoot
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $start
    try {
        if (-not $process.Start()) { throw "Could not start $exe" }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        $timedOut = $false
        while (-not $process.WaitForExit(5000)) {
            Write-Host ("[{0}] {1:N0} s (limit {2} s)" -f $Name, $timer.Elapsed.TotalSeconds, $TimeoutSeconds)
            if ($timer.Elapsed.TotalSeconds -ge $TimeoutSeconds) {
                $timedOut = $true
                $script:stopAfterTimeout = $true
                # Stop only this diagnostic child, never an existing wallet worker.
                if (-not $process.HasExited) { $process.Kill() }
                if (-not $process.WaitForExit(5000)) {
                    throw "Diagnostic child could not be stopped; no more tests will run. Logs: $logDir"
                }
                break
            }
        }
        $status = if ($timedOut) { "TIMEOUT" } elseif ($process.ExitCode -eq 0) { "PASS" } else {
            "FAIL (exit $($process.ExitCode))"
        }
        $captures = @(@{ Task = $stdoutTask; Path = (Join-Path $logDir "$Name.stdout.txt") },
                      @{ Task = $stderrTask; Path = (Join-Path $logDir "$Name.stderr.txt") })
        $output = ""
        foreach ($capture in $captures) {
            if ($capture.Task.Wait(5000)) {
                Set-Content -LiteralPath $capture.Path -Value $capture.Task.Result -Encoding UTF8
                $output += $capture.Task.Result + "`n"
            } else {
                $status = "FAIL (log capture timeout)"
            }
        }
        $speed = $null
        if ($status -eq "PASS") {
            $opencl = [regex]::Match($output, 'wall [0-9.]+ s, wall speed ([0-9.]+) M/s')
            $vulkan = [regex]::Match($output, 'wall: [0-9]+ keys / [0-9.]+ s, ([0-9.]+) (M/s|K/s|/s)')
            if ($opencl.Success) {
                $speed = [double]::Parse($opencl.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
            } elseif ($vulkan.Success) {
                $speed = [double]::Parse($vulkan.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
                if ($vulkan.Groups[2].Value -eq "K/s") { $speed /= 1000.0 }
                if ($vulkan.Groups[2].Value -eq "/s") { $speed /= 1000000.0 }
            } elseif ($Name -ne "01-vulkan-selftest") {
                $status = "FAIL (wall rate missing)"
            }
            if ($Name -ne "01-vulkan-selftest" -and $null -ne $speed -and $speed -le 0) {
                $status = "FAIL (zero wall rate)"
            }
        }
        foreach ($line in ($output -split '\r?\n')) {
            if ($status -ne "PASS" -or $line -match '^(Vulkan resident|Vulkan full-address|OpenCL resident profile|curve batch:|memory:|wall[: ]|GPU stages|GPU stage time|  (curve|affine|keccak|checksum|base58|match)[: ])') {
                if ($line) { Write-Report $line }
            }
        }
        $results.Add([pscustomobject]@{ Test = $Name; Result = $status;
            Seconds = [math]::Round($timer.Elapsed.TotalSeconds, 1);
            MKeysPerSecond = $speed; Arguments = $Arguments -join " " })
        Write-Report ("[$Name] $status")
        return $status -eq "PASS"
    } finally { $process.Dispose() }
}

function Write-Summary {
    Write-Report ($results | Select-Object Test, Result, Seconds, MKeysPerSecond |
                  Format-Table -AutoSize | Out-String)
    $csv = Join-Path $logDir "benchmark.csv"
    $results | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
    Write-Report "Compare wall M/s only; GPU-stage rates exclude host work. Repeats expose drift."
    Write-Report "Machine-readable benchmark: $csv"
    Write-Report "Send summary.txt and benchmark.csv from: $logDir"
}

Write-Report "No wallets or private keys are printed or saved by these tests."
Write-Report "Dictionary SHA-256: $((Get-FileHash -LiteralPath $words -Algorithm SHA256).Hash)"
Write-Report "Order: Vulkan correctness; OpenCL / Vulkan 1 / Vulkan 4 / Vulkan 1 / Vulkan 4 / OpenCL."
if (-not (Invoke-Bounded "01-vulkan-selftest" @("--no-config", "test-vulkan"))) {
    Write-Report "Vulkan correctness failed; no throughput comparison is trustworthy."
    Write-Summary
    exit 1
}

$secondsToken = "$Seconds"
$opencl = @("--no-config", "--backend", "opencl", "--gpu-group-size", "64",
            "--opencl-compiler", "compact", "--opencl-pipeline", "staged",
            "--opencl-inverse", "pair", "--opencl-affine-batch", "4",
            "--opencl-curve-batch", "2", "--opencl-opt-mask", "63",
            "--opencl-profile", "--words", "words.txt", "--gpu-buffer-mb", "8",
            "--bench-seconds", $secondsToken)
$vulkan1 = @("--no-config", "--backend", "vulkan", "--words", "words.txt",
             "--vulkan-profile", "--vulkan-curve-batch", "1", "--vulkan-batch-keys",
             "$VulkanBatchKeys", "--bench-seconds", $secondsToken)
$vulkan4 = @("--no-config", "--backend", "vulkan", "--words", "words.txt",
             "--vulkan-profile", "--vulkan-curve-batch", "4", "--vulkan-batch-keys",
             "$VulkanBatchKeys", "--bench-seconds", $secondsToken)
$allPassed = $true
foreach ($run in @(
    @{ Name = "02-opencl-A"; Args = $opencl },
    @{ Name = "03-vulkan-1-A"; Args = $vulkan1 },
    @{ Name = "04-vulkan-4-A"; Args = $vulkan4 },
    @{ Name = "05-vulkan-1-B"; Args = $vulkan1 },
    @{ Name = "06-vulkan-4-B"; Args = $vulkan4 },
    @{ Name = "07-opencl-B"; Args = $opencl }
)) {
    if (-not (Invoke-Bounded $run.Name $run.Args)) { $allPassed = $false }
    if ($stopAfterTimeout) {
        Write-Report "A child timed out; stopping remaining GPU tests."
        break
    }
}
Write-Summary
if (-not $allPassed) { exit 1 }

# RX 9070 XT comparison and safe backend winner selection. Only no-wallet
# self-test/profile commands run; config updates are opt-in at the script level.
param(
    [ValidateRange(30, 600)][int]$TimeoutSeconds = 120,
    [ValidateRange(1, 60)][int]$Seconds = 5,
    [ValidateSet(32768, 65536, 131072, 262144, 524288, 1048576)][int]$VulkanBatchKeys = 131072,
    [ValidateSet(4, 8)][int]$VulkanAffineBatch = 8,
    [ValidateSet(4, 8, 16)][int]$VulkanResidentGroup = 16,
    [switch]$UpdateConfig
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

function Get-ArgumentValue([string[]]$Arguments, [string]$Name, [string]$Default = "") {
    $index = [array]::IndexOf($Arguments, $Name)
    if ($index -ge 0 -and $index + 1 -lt $Arguments.Count) { return $Arguments[$index + 1] }
    return $Default
}

function Has-Argument([string[]]$Arguments, [string]$Name) {
    return [array]::IndexOf($Arguments, $Name) -ge 0
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
            $vulkanProfile = [regex]::Match($output, 'wall: [0-9]+ keys / [0-9.]+ s, ([0-9.]+) (M/s|K/s|/s)')
            $vulkanResident = [regex]::Match($output, 'Vulkan batch [0-9]+ / affine [0-9]+ / .*?\s+([0-9.]+) (M/s|K/s|/s)')
            if ($opencl.Success) {
                $speed = [double]::Parse($opencl.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
            } elseif ($vulkanProfile.Success -or $vulkanResident.Success) {
                $match = if ($vulkanResident.Success) { $vulkanResident } else { $vulkanProfile }
                $speed = [double]::Parse($match.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
                if ($match.Groups[2].Value -eq "K/s") { $speed /= 1000.0 }
                if ($match.Groups[2].Value -eq "/s") { $speed /= 1000000.0 }
            } elseif ($Name -ne "01-vulkan-selftest") {
                $status = "FAIL (wall rate missing)"
            }
            if ($Name -ne "01-vulkan-selftest" -and $null -ne $speed -and $speed -le 0) {
                $status = "FAIL (zero wall rate)"
            }
        }
        foreach ($line in ($output -split '\r?\n')) {
            if ($status -ne "PASS" -or $line -match '^(Vulkan batch|Vulkan resident|Vulkan full-address|OpenCL resident profile|curve batch:|affine batch:|submit batch:|resident group:|memory:|GPU stage buffers:|wall[: ]|GPU stage dispatches:|queue submits:|GPU stages|GPU stage time|Host wall intervals|candidate records|Benchmark complete|  (curve|affine|keccak|checksum|base58|match|setup|record|submit|fence wait|collect)[: ])') {
                if ($line) { Write-Report $line }
            }
        }
        $results.Add([pscustomobject]@{ Test = $Name; Backend = Get-ArgumentValue $Arguments "--backend" "unknown"; Result = $status;
            Seconds = [math]::Round($timer.Elapsed.TotalSeconds, 1);
            MKeysPerSecond = $speed; Arguments = $Arguments -join " " })
        Write-Report ("[$Name] $status")
        return $status -eq "PASS"
    } finally { $process.Dispose() }
}

function Update-SearchConfig([object]$Best) {
    $configPath = Join-Path $PSScriptRoot "tron-vanity.conf"
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        Write-Report "Config update skipped: $configPath was not found."
        return $false
    }
    $tokens = @($Best.Arguments -split ' ' | Where-Object { $_ -ne "" })
    $backend = [string]$Best.Backend
    if ($backend -eq "vulkan") {
        $updates = [ordered]@{
            "backend" = "vulkan"
            "strict-backend" = "true"
            "seconds" = "0"
            "words" = "words.txt"
            "out" = "results"
            "vulkan-curve-batch" = Get-ArgumentValue $tokens "--vulkan-curve-batch" "4"
            "vulkan-affine-batch" = Get-ArgumentValue $tokens "--vulkan-affine-batch" "8"
            "vulkan-batch-keys" = Get-ArgumentValue $tokens "--vulkan-batch-keys" "131072"
            "vulkan-resident-group" = Get-ArgumentValue $tokens "--vulkan-resident-group" "16"
            "vulkan-field" = Get-ArgumentValue $tokens "--vulkan-field" "10x26"
        }
    } elseif ($backend -eq "opencl") {
        $updates = [ordered]@{
            "backend" = "opencl"
            "strict-backend" = "true"
            "seconds" = "0"
            "words" = "words.txt"
            "out" = "results"
            "gpu-resident" = "true"
            "gpu-rng" = Get-ArgumentValue $tokens "--gpu-rng" "chacha12"
            "gpu-buffer-mb" = Get-ArgumentValue $tokens "--gpu-buffer-mb" "8"
            "gpu-chunk-ms" = Get-ArgumentValue $tokens "--gpu-chunk-ms" "32"
            "gpu-group-size" = Get-ArgumentValue $tokens "--gpu-group-size" "64"
            "opencl-pipeline" = Get-ArgumentValue $tokens "--opencl-pipeline" "staged"
            "opencl-inverse" = Get-ArgumentValue $tokens "--opencl-inverse" "pair"
            "opencl-affine-batch" = Get-ArgumentValue $tokens "--opencl-affine-batch" "4"
            "opencl-curve-batch" = Get-ArgumentValue $tokens "--opencl-curve-batch" "2"
            "opencl-compiler" = Get-ArgumentValue $tokens "--opencl-compiler" "compact"
            "opencl-opt-mask" = Get-ArgumentValue $tokens "--opencl-opt-mask" "63"
            "opencl-async-meta-read" = if (Has-Argument $tokens "--opencl-async-meta-read") { "true" } else { "false" }
            "opencl-sha-ring" = if (Has-Argument $tokens "--opencl-sha-ring") { "true" } else { "false" }
            "opencl-host-seed" = if (Has-Argument $tokens "--opencl-host-seed") { "true" } else { "false" }
        }
    } else {
        Write-Report "Config update skipped: unsupported benchmark backend '$backend'."
        return $false
    }
    $managed = @("backend", "strict-backend", "seconds", "words", "out", "gpu-resident",
        "gpu-rng", "gpu-buffer-mb", "gpu-chunk-ms", "gpu-group-size", "opencl-pipeline",
        "opencl-inverse", "opencl-affine-batch", "opencl-curve-batch", "opencl-compiler",
        "opencl-opt-mask", "opencl-async-meta-read", "opencl-sha-ring", "opencl-host-seed",
        "vulkan-curve-batch", "vulkan-affine-batch", "vulkan-batch-keys", "vulkan-resident-group",
        "vulkan-field")
    $lines = @(Get-Content -LiteralPath $configPath -Encoding UTF8)
    $output = [System.Collections.Generic.List[string]]::new()
    $seen = @{}
    foreach ($line in $lines) {
        if ($line -match '^\s*([A-Za-z0-9-]+)\s*=') {
            $key = $matches[1]
            if ($managed -contains $key) {
                if ($updates.Contains($key)) {
                    $output.Add("$key=$($updates[$key])")
                    $seen[$key] = $true
                }
                continue
            }
        }
        $output.Add($line)
    }
    foreach ($key in $updates.Keys) {
        if (-not $seen.ContainsKey($key)) { $output.Add("$key=$($updates[$key])") }
    }
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $backupPath = "$configPath.bak-$stamp"
    $tempPath = "$configPath.tmp-$([guid]::NewGuid().ToString('N'))"
    try {
        Copy-Item -LiteralPath $configPath -Destination $backupPath -ErrorAction Stop
        Set-Content -LiteralPath $tempPath -Value $output.ToArray() -Encoding UTF8 -ErrorAction Stop
        Move-Item -LiteralPath $tempPath -Destination $configPath -Force -ErrorAction Stop
        Write-Report ("Config updated from benchmark winner: {0} / {1:N3} M/s" -f $Best.Test, $Best.MKeysPerSecond)
        Write-Report "Config backend: $backend"
        Write-Report "Config backup: $backupPath"
        return $true
    } catch {
        Write-Report "Config update failed; existing config was left in place: $($_.Exception.Message)"
        return $false
    } finally {
        if (Test-Path -LiteralPath $tempPath) { Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue }
    }
}

function Write-Summary {
    Write-Report ($results | Select-Object Test, Backend, Result, Seconds, MKeysPerSecond |
                  Format-Table -AutoSize | Out-String)
    $csv = Join-Path $logDir "benchmark.csv"
    $results | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
    $ranked = @($results | Where-Object { $_.Result -eq "PASS" -and $null -ne $_.MKeysPerSecond } |
                          Sort-Object -Property MKeysPerSecond -Descending)
    if ($ranked.Count) {
        Write-Report "Ranked full-address wall throughput:"
        Write-Report ($ranked | Select-Object Test, Backend, MKeysPerSecond | Format-Table -AutoSize | Out-String)
        Write-Report ("Fastest measured: " + $ranked[0].Test + " / " + $ranked[0].Backend +
                      " (" + $ranked[0].MKeysPerSecond + " M/s)")
        if ($UpdateConfig) { [void](Update-SearchConfig $ranked[0]) }
    } elseif ($UpdateConfig) {
        Write-Report "Config update skipped: no successful timed profile was available."
    }
    Write-Report "Compare wall M/s only; GPU-stage rates exclude host work. Repeats expose drift."
    Write-Report "Machine-readable benchmark: $csv"
    Write-Report "Send summary.txt and benchmark.csv from: $logDir"
}

Write-Report "No wallets or private keys are printed or saved by these tests."
Write-Report "Executable SHA-256: $((Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash)"
Write-Report "Dictionary SHA-256: $((Get-FileHash -LiteralPath $words -Algorithm SHA256).Hash)"
Write-Report "Order: Vulkan correctness; OpenCL reference; Vulkan baseline; Vulkan curve4/affine4; Vulkan winner curve4/affine8 twice; OpenCL reference."
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
$vulkanBaseline = @("--no-config", "--backend", "vulkan", "--words", "words.txt",
             "--bench-resident", "--vulkan-curve-batch", "1", "--vulkan-affine-batch", "4", "--vulkan-resident-group", "8", "--vulkan-batch-keys",
             "$VulkanBatchKeys", "--bench-seconds", $secondsToken)
$vulkanCurve4 = @("--no-config", "--backend", "vulkan", "--words", "words.txt",
             "--bench-resident", "--vulkan-curve-batch", "4", "--vulkan-affine-batch", "4", "--vulkan-resident-group", "$VulkanResidentGroup", "--vulkan-batch-keys",
             "$VulkanBatchKeys", "--bench-seconds", $secondsToken)
$vulkanWinner = @("--no-config", "--backend", "vulkan", "--words", "words.txt",
             "--bench-resident", "--vulkan-curve-batch", "4", "--vulkan-affine-batch", "$VulkanAffineBatch", "--vulkan-resident-group", "$VulkanResidentGroup", "--vulkan-batch-keys",
             "$VulkanBatchKeys", "--bench-seconds", $secondsToken)
$allPassed = $true
foreach ($run in @(
    @{ Name = "02-opencl-A"; Args = $opencl },
    @{ Name = "03-vulkan-baseline-A"; Args = $vulkanBaseline },
    @{ Name = "04-vulkan-curve4-affine4-A"; Args = $vulkanCurve4 },
    @{ Name = "05-vulkan-winner-A"; Args = $vulkanWinner },
    @{ Name = "06-vulkan-winner-B"; Args = $vulkanWinner },
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

# Diagnostics only: no wallet files or private keys, no driver/cache changes.
param(
    [ValidateSet("single", "pair")][string]$Inverse = "pair",
    [ValidateSet("compact", "default")][string]$Compiler = "compact",
    [ValidateSet("staged", "monolithic")][string]$Pipeline = "staged",
    [ValidateRange(30, 600)][int]$TimeoutSeconds = 30,
    [switch]$CompareStages,
    [switch]$CompareAffineBatches
)
$ErrorActionPreference = "Stop"
if ($CompareStages -and $Pipeline -ne "staged") { throw "-CompareStages requires -Pipeline staged." }
if ($CompareAffineBatches -and $Pipeline -ne "staged") { throw "-CompareAffineBatches requires -Pipeline staged." }
$exe = Join-Path $PSScriptRoot "tron_vanity_generator.exe"
if (-not (Test-Path $exe)) { throw "Extract the release ZIP before running this script." }

$logDir = Join-Path $PSScriptRoot ("opencl-diagnostic-" + (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + [guid]::NewGuid().ToString("N").Substring(0, 6))
New-Item -ItemType Directory -Path $logDir | Out-Null
$summary = Join-Path $logDir "summary.txt"
$results = [System.Collections.Generic.List[object]]::new()
Write-Host "Diagnostics: $logDir"

function Write-Report([string]$Text) {
    Write-Host $Text
    Add-Content -LiteralPath $summary -Value $Text -Encoding UTF8
}

function Invoke-BoundedTest([string]$Name, [string[]]$TestArguments, [bool]$CompactReport = $false) {
    Write-Report ("`n[" + $Name + "] " + ($TestArguments -join " "))
    $stdoutPath = Join-Path $logDir "$Name.stdout.txt"
    $stderrPath = Join-Path $logDir "$Name.stderr.txt"
    # Keep the original process handle. Start-Process -PassThru on Windows
    # PowerShell 5.1 can return a null ExitCode after WaitForExit.
    $start = [System.Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $exe
    # All arguments here are fixed CLI tokens/validated choices, without spaces.
    $start.Arguments = $TestArguments -join " "
    $start.WorkingDirectory = $PSScriptRoot
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $start
    if (-not $process.Start()) { throw "Could not start diagnostic child: $exe" }
    # Drain both pipes concurrently so a verbose compiler cannot block on a
    # full pipe while the launcher waits for the process to finish.
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    $timedOut = $false
    while (-not $process.WaitForExit(5000)) {
        Write-Host ("[{0}] {1:N0} s (limit {2} s)" -f $Name, $timer.Elapsed.TotalSeconds, $TimeoutSeconds)
        if ($timer.Elapsed.TotalSeconds -ge $TimeoutSeconds) {
            $timedOut = $true
            # Never kill another worker, an ICD service, or the GPU driver.
            if (-not $process.HasExited) { $process.Kill() }
            if (-not $process.WaitForExit(5000)) {
                throw "The diagnostic child could not be stopped; no more GPU tests will be started. Logs: $logDir"
            }
            break
        }
    }
    $status = if ($timedOut) { "TIMEOUT" } elseif ($process.ExitCode -eq 0) { "PASS" } else { "FAIL (exit $($process.ExitCode))" }
    foreach ($capture in @(@{ Task = $stdoutTask; Path = $stdoutPath }, @{ Task = $stderrTask; Path = $stderrPath })) {
        if ($capture.Task.Wait(5000)) {
            Set-Content -LiteralPath $capture.Path -Value $capture.Task.Result -Encoding UTF8
        } else {
            # Do not hang if some driver-created child inherited a pipe handle.
            Write-Report "Log capture did not finish: $($capture.Path)"
            $status = "FAIL (log capture timeout)"
        }
    }
    foreach ($file in @($stdoutPath, $stderrPath)) {
        if (Test-Path -LiteralPath $file) {
            $text = Get-Content -LiteralPath $file -Raw -Encoding UTF8
            if ($text) {
                if ($CompactReport -and $status -eq "PASS") {
                    foreach ($line in ($text -split '\r?\n')) {
                        if ($line -match '^(gfx\d+:|wall |base preparation |Driver-reported GPU|GPU stage time|  (curve|affine|keccak|checksum|base58|match) )') {
                            Write-Report $line
                        }
                    }
                } else { Write-Report $text }
            }
        }
    }
    $results.Add([pscustomobject]@{ Test = $Name; Result = $status; Seconds = [math]::Round($timer.Elapsed.TotalSeconds, 1) })
    Write-Report ("[" + $Name + "] " + $status)
    $process.Dispose()
    return $status -eq "PASS"
}

function Write-Summary {
    Write-Report ($results | Format-Table -AutoSize | Out-String)
    Write-Report "Send summary.txt from: $logDir"
}

$common = @("--backend", "opencl", "--gpu-group-size", "64", "--opencl-compiler", $Compiler, "--opencl-pipeline", $Pipeline)
$smokeOk = Invoke-BoundedTest "01-smoke" ($common + @("--opencl-diagnose", "smoke"))
if (-not $smokeOk) {
    Write-Report "Even the tiny OpenCL kernel failed/timed out. EC/RNG optimization is not isolated as the cause. Stopping."
    Write-Summary
    exit 1
}
$rngOk = Invoke-BoundedTest "02-rng-only" ($common + @("--opencl-diagnose", "rng"))
$singleBuildOk = $true
$pairBuildOk = $true
if ($Pipeline -eq "staged") {
    $buildsOk = $true
    foreach ($stage in @("curve", "affine-single", "affine-pair", "keccak", "checksum", "base58", "match")) {
        # Each program gets a fresh bounded child, including after another
        # stage times out. These are BUILD checks, not execution self-tests.
        if ($stage.StartsWith("affine-")) {
            $mode = $stage.Substring(7)
            $built = Invoke-BoundedTest "02-build-$stage" ($common + @("--opencl-diagnose", "build-affine", "--opencl-inverse", $mode))
            if ($mode -eq "single") { $singleBuildOk = $built } else { $pairBuildOk = $built }
        } else {
            $built = Invoke-BoundedTest "02-build-$stage" ($common + @("--opencl-diagnose", "build-$stage"))
            if (-not $built) { $buildsOk = $false }
        }
    }
    if (-not $buildsOk -or (-not $singleBuildOk -and -not $pairBuildOk)) {
        Write-Report "At least one staged program did not build. No scan/profile or real search will be launched."
        Write-Summary
        exit 1
    }
}
$singleOk = $false
$pairOk = $false
if ($singleBuildOk) { $singleOk = Invoke-BoundedTest "03-scan-single" ($common + @("--opencl-diagnose", "scan", "--opencl-inverse", "single")) }
else { Write-Report "Single scan skipped: its affine program did not build." }
if ($pairBuildOk) { $pairOk = Invoke-BoundedTest "04-scan-pair" ($common + @("--opencl-diagnose", "scan", "--opencl-inverse", "pair")) }
else { Write-Report "Paired scan skipped: its affine program did not build." }
$selected = if ($Inverse -eq "single" -and $singleOk) { "single" } elseif ($pairOk) { "pair" } elseif ($singleOk) { "single" } else { "" }
if (-not $selected) {
    Write-Report "Both scan-only variants failed/timed out. No working resident mode confirmed."
    Write-Summary
    exit 1
}
$fullOk = $false
if ($rngOk) {
    $fullOk = Invoke-BoundedTest "05-combined" ($common + @("--opencl-diagnose", "full", "--opencl-inverse", $selected))
}
$selectedArgs = $common + @("--opencl-inverse", $selected)
if (-not $fullOk) {
    $selectedArgs += "--opencl-host-seed"
    Write-Report "Using the validated scan-only path with OS CSPRNG. GPU address search is unchanged; GPU RNG compilation is excluded."
}
if (Test-Path (Join-Path $PSScriptRoot "words.txt")) {
    $profileOk = Invoke-BoundedTest "06-profile" ($selectedArgs + @("--opencl-profile", "--words", "words.txt", "--gpu-buffer-mb", "8", "--bench-seconds", "5"))
    if (-not $profileOk) { Write-Summary; exit 1 }
    if ($CompareStages) {
        # One optimization bit at a time against the exact v1.7.2 math path.
        # Every child is bounded; no search or wallet output is launched.
        $stageCases = @(
            @{ Name = "baseline"; Mask = "0" },
            @{ Name = "curve"; Mask = "1" },
            @{ Name = "affine"; Mask = "2" },
            @{ Name = "keccak"; Mask = "4" },
            @{ Name = "checksum"; Mask = "8" },
            @{ Name = "base58"; Mask = "16" },
            @{ Name = "match"; Mask = "32" }
        )
        foreach ($case in $stageCases) {
            if ($selected -ne "pair" -and $case.Name -eq "affine") { continue }
            $profileArgs = $selectedArgs + @("--opencl-opt-mask", $case.Mask, "--opencl-profile",
                                             "--words", "words.txt", "--gpu-buffer-mb", "8", "--bench-seconds", "5")
            if (-not (Invoke-BoundedTest ("07-" + $case.Name) $profileArgs $true)) {
                Write-Summary
                exit 1
            }
        }
    }
    if ($CompareAffineBatches) {
        if ($selected -ne "pair") {
            Write-Report "Affine batch comparison skipped: paired inversion did not pass the scan self-test."
        } else {
            foreach ($batch in @(2, 4, 8)) {
                $profileArgs = $selectedArgs + @("--opencl-affine-batch", "$batch", "--opencl-profile",
                                                 "--words", "words.txt", "--gpu-buffer-mb", "8", "--bench-seconds", "5")
                if (-not (Invoke-BoundedTest ("08-affine-$batch") $profileArgs $true)) {
                    Write-Summary
                    exit 1
                }
            }
        }
    }
} else { Write-Report "words.txt not found: profile skipped, self-tests did not need a dictionary." }
Write-Report ("Self-test passed. Optional search command (NOT executed):`ntron_vanity_generator.exe " + (($selectedArgs + @("--gpu-resident", "--words", "words.txt", "--seconds", "60")) -join " "))
Write-Summary

# Diagnostics only: no wallet files or private keys, no driver/cache changes.
# With -UpdateConfig, the existing search config is backed up and updated only
# after the selected benchmark suite has a valid winning profile.
param(
    [ValidateSet("single", "pair")][string]$Inverse = "pair",
    [ValidateSet("compact", "default")][string]$Compiler = "compact",
    [ValidateSet("staged", "monolithic")][string]$Pipeline = "staged",
    [ValidateRange(30, 600)][int]$TimeoutSeconds = 30,
    [switch]$CompareStages,
    [switch]$CompareAffineBatches,
    [switch]$CompareCurveBatches,
    [switch]$CompareShaRing,
    [switch]$CompareGroupSizes,
    [switch]$CompareMetaRead,
    [switch]$CompareRuntimeSizing,
    [switch]$UpdateConfig,
    [switch]$All
)
$ErrorActionPreference = "Stop"
if ($All) {
    $CompareStages = $true
    $CompareAffineBatches = $true
    $CompareCurveBatches = $true
    $CompareShaRing = $true
    $CompareGroupSizes = $true
    $CompareMetaRead = $true
    $CompareRuntimeSizing = $true
}
if ($CompareStages -and $Pipeline -ne "staged") { throw "-CompareStages requires -Pipeline staged." }
if ($CompareAffineBatches -and $Pipeline -ne "staged") { throw "-CompareAffineBatches requires -Pipeline staged." }
if ($CompareCurveBatches -and $Pipeline -ne "staged") { throw "-CompareCurveBatches requires -Pipeline staged." }
if ($CompareShaRing -and $Pipeline -ne "staged") { throw "-CompareShaRing requires -Pipeline staged." }
if ($CompareGroupSizes -and $Pipeline -ne "staged") { throw "-CompareGroupSizes requires -Pipeline staged." }
if ($CompareMetaRead -and $Pipeline -ne "staged") { throw "-CompareMetaRead requires -Pipeline staged." }
if ($CompareRuntimeSizing -and $Pipeline -ne "staged") { throw "-CompareRuntimeSizing requires -Pipeline staged." }
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

Write-Report "Executable SHA-256: $((Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash)"
$dictionaryPath = Join-Path $PSScriptRoot "words.txt"
if (Test-Path -LiteralPath $dictionaryPath) {
    Write-Report "Dictionary SHA-256: $((Get-FileHash -LiteralPath $dictionaryPath -Algorithm SHA256).Hash)"
}
Write-Report "All diagnostic children use --no-config; the adjacent search config cannot alter benchmark options."

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
    $speed = $null
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
                if ($text -match 'wall [0-9.]+ s, wall speed ([0-9.]+) M/s') {
                    $speed = [double]::Parse($matches[1], [Globalization.CultureInfo]::InvariantCulture)
                }
                if ($CompactReport -and $status -eq "PASS") {
                    foreach ($line in ($text -split '\r?\n')) {
                        if ($line -match '^(gfx\d+:|wall |base preparation |host timing:|Driver-reported GPU|GPU stage time|  (curve|affine|keccak|checksum|base58|match) )') {
                            Write-Report $line
                        }
                    }
                } else { Write-Report $text }
            }
        }
    }
    $results.Add([pscustomobject]@{ Test = $Name; Result = $status; Seconds = [math]::Round($timer.Elapsed.TotalSeconds, 1);
                                     MKeysPerSecond = $speed; Arguments = $TestArguments -join " " })
    Write-Report ("[" + $Name + "] " + $status)
    $process.Dispose()
    return $status -eq "PASS"
}

function Get-ArgumentValue([string[]]$Arguments, [string]$Name, [string]$Default = "") {
    $index = [array]::IndexOf($Arguments, $Name)
    if ($index -ge 0 -and $index + 1 -lt $Arguments.Count) { return $Arguments[$index + 1] }
    return $Default
}

function Has-Argument([string[]]$Arguments, [string]$Name) {
    return [array]::IndexOf($Arguments, $Name) -ge 0
}

function Update-SearchConfig([object]$Best) {
    $configPath = Join-Path $PSScriptRoot "tron-vanity.conf"
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        Write-Report "Config update skipped: $configPath was not found."
        return $false
    }
    $tokens = @($Best.Arguments -split ' ' | Where-Object { $_ -ne "" })
    $updates = [ordered]@{
        "backend" = "opencl"
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
    $lines = @(Get-Content -LiteralPath $configPath -Encoding UTF8)
    $output = [System.Collections.Generic.List[string]]::new()
    $seen = @{}
    foreach ($line in $lines) {
        if ($line -match '^(\s*)([A-Za-z0-9-]+)(\s*=).*$') {
            $key = $matches[2]
            if ($updates.Contains($key)) {
                $output.Add(($matches[1] + $key + $matches[3] + [string]$updates[$key]))
                $seen[$key] = $true
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
        Write-Report ("Config updated from benchmark winner: {0} ({1:N3} M/s)" -f $Best.Test, $Best.MKeysPerSecond)
        Write-Report "Config backup: $backupPath"
        Write-Report ((($updates.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ", "))
        return $true
    } catch {
        Write-Report "Config update failed; existing config was left in place: $($_.Exception.Message)"
        return $false
    } finally {
        if (Test-Path -LiteralPath $tempPath) { Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue }
    }
}

function Write-Summary {
    Write-Report ($results | Select-Object Test, Result, Seconds, MKeysPerSecond | Format-Table -AutoSize | Out-String)
    $csv = Join-Path $logDir "benchmark.csv"
    $results | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
    $ranked = @($results | Where-Object { $_.Result -eq "PASS" -and $null -ne $_.MKeysPerSecond } |
                          Sort-Object -Property MKeysPerSecond -Descending)
    if ($ranked.Count) {
        Write-Report "Ranked wall throughput (same dictionary and five-second workload):"
        Write-Report ($ranked | Select-Object Test, MKeysPerSecond | Format-Table -AutoSize | Out-String)
        Write-Report ("Fastest measured: " + $ranked[0].Test + " (" + $ranked[0].MKeysPerSecond + " M/s)")
        # Optional comparison cases are allowed to fail or time out on a
        # particular driver.  They must not prevent a separately verified
        # passing profile from becoming the default.  Every row in $ranked
        # already passed its own GPU/CPU correctness gate when -All is used,
        # and only rows with a parsed positive wall rate can enter it.
        if ($UpdateConfig) {
            [void](Update-SearchConfig $ranked[0])
        }
    } elseif ($UpdateConfig) {
        Write-Report "Config update skipped: no successful timed profile was available."
    }
    Write-Report "Machine-readable benchmark: $csv"
    Write-Report "Send summary.txt from: $logDir"
}

function Invoke-Variant([string]$Name, [string[]]$VariantArguments, [string[]]$BaseArguments,
                        [string]$CheckStage = "scan", [int]$BufferMiB = 8,
                        [int]$ChunkMs = 32) {
    if (-not $BaseArguments) { $BaseArguments = $selectedArgs }
    # Keep the correctness gate and timed profile on the same ring/chunk
    # settings. The packaged RX config uses 128 MiB; earlier profiles used 8.
    $variant = @($BaseArguments) + @($VariantArguments) + @("--gpu-buffer-mb", "$BufferMiB", "--gpu-chunk-ms", "$ChunkMs")
    if ($All) {
        # Do not profile a configuration that cannot reproduce CPU-verified
        # GPU addresses/scalars. Each check is a separate bounded process.
        if (-not (Invoke-BoundedTest ($Name + "-check") ($variant + @("--opencl-diagnose", $CheckStage)))) {
            return $false
        }
    }
    $profileArgs = $variant + @("--opencl-profile", "--words", "words.txt",
                               "--bench-seconds", "5")
    return Invoke-BoundedTest $Name $profileArgs $true
}

$common = @("--no-config", "--backend", "opencl", "--gpu-group-size", "64", "--opencl-compiler", $Compiler, "--opencl-pipeline", $Pipeline)
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
            if (-not (Invoke-Variant -Name ("07-" + $case.Name) -VariantArguments @("--opencl-opt-mask", $case.Mask) -BaseArguments $selectedArgs)) {
                if (-not $All) { Write-Summary; exit 1 }
            }
        }
    }
    if ($CompareAffineBatches) {
        if ($selected -ne "pair") {
            Write-Report "Affine batch comparison skipped: paired inversion did not pass the scan self-test."
        } else {
            foreach ($batch in @(2, 4, 8)) {
                if (-not (Invoke-Variant -Name ("08-affine-$batch") -VariantArguments @("--opencl-affine-batch", "$batch") -BaseArguments $selectedArgs)) {
                    if (-not $All) { Write-Summary; exit 1 }
                }
            }
        }
    }
    if ($CompareCurveBatches) {
        foreach ($batch in @(2, 4, 8)) {
            if (-not (Invoke-Variant -Name ("09-curve-$batch") -VariantArguments @("--opencl-curve-batch", "$batch") -BaseArguments $selectedArgs)) {
                if (-not $All) { Write-Summary; exit 1 }
            }
        }
    }
    if ($CompareShaRing) {
        foreach ($case in @(@{ Name = "default"; Ring = $false }, @{ Name = "ring"; Ring = $true })) {
            $variant = if ($case.Ring) { @("--opencl-sha-ring") } else { @() }
            if (-not (Invoke-Variant -Name ("10-sha-" + $case.Name) -VariantArguments $variant -BaseArguments $selectedArgs)) {
                if (-not $All) { Write-Summary; exit 1 }
            }
        }
    }
    if ($CompareGroupSizes) {
        foreach ($size in @(64, 128, 256)) {
            $groupArgs = @($selectedArgs)
            $groupIndex = [array]::IndexOf($groupArgs, "--gpu-group-size")
            if ($groupIndex -lt 0) { throw "Internal error: missing --gpu-group-size in diagnostic arguments." }
            $groupArgs[$groupIndex + 1] = "$size"
            if ($size -ne 64 -and -not $All) {
                # A changed local size must pass the full GPU/CPU address and
                # private-scalar self-test before its throughput is profiled.
                if (-not (Invoke-BoundedTest ("11-group-$size-scan") ($groupArgs + @("--opencl-diagnose", "scan")))) {
                    Write-Summary
                    exit 1
                }
            }
            if (-not (Invoke-Variant -Name ("11-group-$size-profile") -VariantArguments @() -BaseArguments $groupArgs)) {
                if (-not $All) { Write-Summary; exit 1 }
            }
        }
    }
    if ($CompareMetaRead) {
        $profileArgs = $selectedArgs + @("--opencl-profile", "--words", "words.txt",
                                         "--gpu-buffer-mb", "8", "--bench-seconds", "5")
        if (-not (Invoke-BoundedTest "12-meta-blocking" $profileArgs $true)) {
            if (-not $All) { Write-Summary; exit 1 }
        }
        # The queued read must preserve GPU/CPU address and scalar agreement
        # before its throughput is compared. This never writes wallets.
        if (-not (Invoke-BoundedTest "12-meta-queued-scan" ($selectedArgs + @("--opencl-async-meta-read", "--opencl-diagnose", "scan")))) {
            if (-not $All) { Write-Summary; exit 1 }
        } elseif (-not (Invoke-BoundedTest "12-meta-queued" ($profileArgs + "--opencl-async-meta-read") $true)) {
            if (-not $All) { Write-Summary; exit 1 }
        }
    }
    if ($All) {
        # Combined candidates catch interactions that one-knob-at-a-time
        # comparisons miss. The changed group and math are verified together.
        $group128 = @($selectedArgs)
        $groupIndex = [array]::IndexOf($group128, "--gpu-group-size")
        $group128[$groupIndex + 1] = "128"
        foreach ($variant in @(
            @{ Name = "affine-2"; Args = @("--opencl-affine-batch", "2") },
            @{ Name = "affine-8"; Args = @("--opencl-affine-batch", "8") },
            @{ Name = "curve-4"; Args = @("--opencl-curve-batch", "4") },
            @{ Name = "queued"; Args = @("--opencl-async-meta-read") }
        )) {
            if (-not (Invoke-Variant -Name ("13-group-128-" + $variant.Name) -VariantArguments $variant.Args -BaseArguments $group128)) {
                Write-Report ("Variant failed; continuing: " + $variant.Name)
            }
        }
        if ($fullOk) {
            foreach ($rng in @("aes-ctr", "philox")) {
                if (-not (Invoke-Variant -Name ("14-rng-" + $rng) -VariantArguments @("--gpu-rng", $rng) -BaseArguments $selectedArgs -CheckStage "full")) {
                    Write-Report ("RNG variant failed; continuing: " + $rng)
                }
            }
        } else { Write-Report "GPU RNG variants skipped: combined GPU RNG self-test did not pass." }
    }
    if ($CompareRuntimeSizing) {
        # The 8/128/8 sequence isolates the release config's 128 MiB ring
        # from drift. Chunk comparisons then use that same release ring size.
        foreach ($case in @(
            @{ Name = "15-buffer-8-A"; Buffer = 8; Chunk = 32 },
            @{ Name = "15-buffer-128"; Buffer = 128; Chunk = 32 },
            @{ Name = "15-buffer-8-B"; Buffer = 8; Chunk = 32 },
            @{ Name = "16-chunk-16"; Buffer = 128; Chunk = 16 },
            @{ Name = "16-chunk-32"; Buffer = 128; Chunk = 32 },
            @{ Name = "16-chunk-64"; Buffer = 128; Chunk = 64 }
        )) {
            $parameters = @{ Name = $case.Name; VariantArguments = @();
                             BaseArguments = $selectedArgs; BufferMiB = $case.Buffer;
                             ChunkMs = $case.Chunk }
            if (-not (Invoke-Variant @parameters)) {
                if (-not $All) { Write-Summary; exit 1 }
            }
        }
    }
} else { Write-Report "words.txt not found: profile skipped, self-tests did not need a dictionary." }
Write-Report ("Base self-test passed; check the table for any failed optional variant. Optional search command (NOT executed):`ntron_vanity_generator.exe " + (($selectedArgs + @("--gpu-resident", "--words", "words.txt", "--seconds", "60")) -join " "))
Write-Summary
if ($All -and @($results | Where-Object { $_.Result -ne "PASS" }).Count) { exit 1 }

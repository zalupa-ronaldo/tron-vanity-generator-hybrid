# Run with Windows PowerShell 5.1, matching test-opencl.cmd. No GPU required:
# the fixture is a real child process, so redirection/exit/timeout paths are tested.
$ErrorActionPreference = "Stop"
$root = Join-Path ([IO.Path]::GetTempPath()) ("tron-launcher-test-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $root | Out-Null
$fixture = Join-Path $root "fixture.exe"
Add-Type -OutputAssembly $fixture -OutputType ConsoleApplication -TypeDefinition @'
using System;
using System.IO;
using System.Threading;
public class Fixture {
    static string Value(string[] args, string name) {
        int i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : "";
    }
    public static int Main(string[] args) {
        string mode = Environment.GetEnvironmentVariable("TRON_LAUNCHER_FIXTURE");
        string stage = Value(args, "--opencl-diagnose");
        if (Array.IndexOf(args, "--opencl-profile") >= 0) stage = "profile";
        string name = stage == "scan" ? "scan-" + Value(args, "--opencl-inverse") : stage;
        if (stage == "build-affine") name += "-" + Value(args, "--opencl-inverse");
        if (stage == "profile" && mode == "compare-stages" && Value(args, "--opencl-opt-mask") != "")
            name += "-mask-" + Value(args, "--opencl-opt-mask");
        if (stage == "profile" && mode == "compare-affine" && Value(args, "--opencl-affine-batch") != "")
            name += "-affine-" + Value(args, "--opencl-affine-batch");
        if (stage == "profile" && mode == "compare-curve" && Value(args, "--opencl-curve-batch") != "")
            name += "-curve-" + Value(args, "--opencl-curve-batch");
        if (stage == "profile" && mode == "compare-sha" && Array.IndexOf(args, "--opencl-sha-ring") >= 0)
            name += "-sha-ring";
        if (mode == "compare-groups" && stage == "profile")
            name += "-group-" + Value(args, "--gpu-group-size");
        if (mode == "compare-groups" && stage == "scan" && Value(args, "--gpu-group-size") != "64")
            name += "-group-" + Value(args, "--gpu-group-size");
        File.AppendAllText("calls.txt", name + Environment.NewLine);
        Console.WriteLine("fixture " + name);
        if (stage == "profile") Console.WriteLine("host timing: enqueue 0.001 s, finish wait 0.002 s, event query 0.003 s, metadata read 0.004 s, records 0.000 s, metadata update 0.000 s");
        Console.Error.WriteLine("OpenCL API: fixture BEGIN");
        Console.Error.Flush();
        if (stage == "profile") {
            bool needsHostSeed = mode == "rng-fail" || mode == "rng-timeout" || mode == "combined-fail";
            if (needsHostSeed != (Array.IndexOf(args, "--opencl-host-seed") >= 0)) return 3;
            if ((mode == "pair-fail" || mode == "affine-pair-fail") && Value(args, "--opencl-inverse") != "single") return 3;
        }
        if (mode == "rng-timeout" && stage == "rng") Thread.Sleep(60000);
        if (mode == "smoke-fail" && stage == "smoke") return 2;
        if (mode == "curve-build-fail" && stage == "build-curve") return 2;
        if (mode == "checksum-build-fail" && stage == "build-checksum") return 2;
        if (mode == "affine-pair-fail" && name == "build-affine-pair") return 2;
        if (mode == "rng-fail" && stage == "rng") return 2;
        if (mode == "pair-fail" && name == "scan-pair") return 2;
        if (mode == "combined-fail" && stage == "full") return 2;
        if (mode == "profile-fail" && stage == "profile") return 2;
        if (mode == "all-scan-fail" && stage == "scan") return 2;
        return 0;
    }
}
'@
$cases = @(
    @{ Mode = "success"; Exit = 0; Calls = "smoke,rng,scan-single,scan-pair,full,profile"; Text = "Self-test passed" },
    @{ Mode = "smoke-fail"; Exit = 1; Calls = "smoke"; Text = "tiny OpenCL kernel failed" },
    @{ Mode = "rng-fail"; Exit = 0; Calls = "smoke,rng,scan-single,scan-pair,profile"; Text = "--opencl-host-seed" },
    @{ Mode = "pair-fail"; Exit = 0; Calls = "smoke,rng,scan-single,scan-pair,full,profile"; Text = "--opencl-inverse single" },
    @{ Mode = "combined-fail"; Exit = 0; Calls = "smoke,rng,scan-single,scan-pair,full,profile"; Text = "--opencl-host-seed" },
    @{ Mode = "all-scan-fail"; Exit = 1; Calls = "smoke,rng,scan-single,scan-pair"; Text = "No working resident mode confirmed" },
    @{ Mode = "profile-fail"; Exit = 1; Calls = "smoke,rng,scan-single,scan-pair,full,profile"; Text = "FAIL (exit 2)" },
    @{ Mode = "rng-timeout"; Exit = 0; Calls = "smoke,rng,scan-single,scan-pair,profile"; Text = "TIMEOUT" }
)
$builds = ",build-curve,build-affine-single,build-affine-pair,build-keccak,build-checksum,build-base58,build-match"
foreach ($case in $cases) { $case.Calls = $case.Calls.Replace(",scan-single", $builds + ",scan-single") }
$cases += @{ Mode = "compare-stages"; CompareStages = $true; Exit = 0; Calls = "smoke,rng" + $builds + ",scan-single,scan-pair,full,profile,profile-mask-0,profile-mask-1,profile-mask-2,profile-mask-4,profile-mask-8,profile-mask-16,profile-mask-32"; Text = "[07-match] PASS" }
$cases += @{ Mode = "compare-affine"; CompareAffine = $true; Exit = 0; Calls = "smoke,rng" + $builds + ",scan-single,scan-pair,full,profile,profile-affine-2,profile-affine-4,profile-affine-8"; Text = "[08-affine-8] PASS" }
$cases += @{ Mode = "compare-curve"; CompareCurve = $true; Exit = 0; Calls = "smoke,rng" + $builds + ",scan-single,scan-pair,full,profile,profile-curve-2,profile-curve-4,profile-curve-8"; Text = "[09-curve-8] PASS" }
$cases += @{ Mode = "compare-sha"; CompareSha = $true; Exit = 0; Calls = "smoke,rng" + $builds + ",scan-single,scan-pair,full,profile,profile,profile-sha-ring"; Text = "[10-sha-ring] PASS" }
$cases += @{ Mode = "compare-groups"; CompareGroups = $true; Exit = 0; Calls = "smoke,rng" + $builds + ",scan-single,scan-pair,full,profile,profile-group-64,scan-pair-group-128,profile-group-128,scan-pair-group-256,profile-group-256"; Text = "[11-group-256-profile] PASS" }
$cases += @{ Mode = "curve-build-fail"; Exit = 1; Calls = "smoke,rng" + $builds; Text = "At least one staged program did not build" }
$cases += @{ Mode = "checksum-build-fail"; Exit = 1; Calls = "smoke,rng" + $builds; Text = "At least one staged program did not build" }
$cases += @{ Mode = "affine-pair-fail"; Exit = 0; Calls = "smoke,rng" + $builds + ",scan-single,full,profile"; Text = "Paired scan skipped" }
$cases += @{ Mode = "monolithic"; Pipeline = "monolithic"; Exit = 0; Calls = "smoke,rng,scan-single,scan-pair,full,profile"; Text = "--opencl-pipeline monolithic" }
$previousMode = $env:TRON_LAUNCHER_FIXTURE
try {
    foreach ($case in $cases) {
        # Include a space in the extraction path to exercise Windows quoting.
        $dir = Join-Path $root ("test " + $case.Mode)
        New-Item -ItemType Directory -Path $dir | Out-Null
        Copy-Item -LiteralPath $fixture -Destination (Join-Path $dir "tron_vanity_generator.exe")
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot "../test-opencl.ps1") -Destination $dir
        Set-Content -LiteralPath (Join-Path $dir "words.txt") -Value "energy" -Encoding ASCII
        $env:TRON_LAUNCHER_FIXTURE = $case.Mode
        $pipeline = if ($case.Pipeline) { $case.Pipeline } else { "staged" }
        $launcherArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $dir "test-opencl.ps1"), "-TimeoutSeconds", "30", "-Pipeline", $pipeline)
        if ($case.CompareStages) { $launcherArgs += "-CompareStages" }
        if ($case.CompareAffine) { $launcherArgs += "-CompareAffineBatches" }
        if ($case.CompareCurve) { $launcherArgs += "-CompareCurveBatches" }
        if ($case.CompareSha) { $launcherArgs += "-CompareShaRing" }
        if ($case.CompareGroups) { $launcherArgs += "-CompareGroupSizes" }
        $output = & powershell.exe @launcherArgs 2>&1
        if ($LASTEXITCODE -ne $case.Exit) { throw "$($case.Mode) exit mismatch: $LASTEXITCODE`n$($output -join "`n")" }
        $calls = (Get-Content -LiteralPath (Join-Path $dir "calls.txt")) -join ","
        if ($calls -ne $case.Calls) { throw "$($case.Mode) stage mismatch: $calls" }
        $reports = @(Get-ChildItem -LiteralPath $dir -Recurse -Filter summary.txt)
        if ($reports.Count -ne 1) { throw "$($case.Mode) missing summary" }
        $report = Get-Content -LiteralPath $reports[0].FullName -Raw
        foreach ($expected in @($case.Text, "OpenCL API: fixture BEGIN", "Send summary.txt")) {
            if (-not $report.Contains($expected)) { throw "$($case.Mode) missing '$expected'`n$report" }
        }
        $expectedTimingLines = if ($case.CompareStages) { 8 } elseif ($case.CompareAffine -or $case.CompareCurve -or $case.CompareGroups) { 4 } elseif ($case.CompareSha) { 3 } elseif ($case.Calls.Contains("profile")) { 1 } else { 0 }
        $timingLines = ([regex]::Matches($report, "host timing:")).Count
        if ($timingLines -ne $expectedTimingLines) { throw "$($case.Mode) host timing summary mismatch: $timingLines instead of $expectedTimingLines" }
        if ($case.Exit -ne 0 -and $report.Contains("Optional search command")) { throw "Failed test suggested a search" }
        if (Get-ChildItem -LiteralPath $dir -Recurse -Filter *.jsonl) { throw "Diagnostic wrote wallet output" }
        Write-Host "Launcher fixture PASS: $($case.Mode)"
    }
} finally {
    $env:TRON_LAUNCHER_FIXTURE = $previousMode
}
# Only remove this test's newly allocated directory after all checks pass.
Remove-Item -LiteralPath $root -Recurse -Force

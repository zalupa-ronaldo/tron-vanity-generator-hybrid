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
        File.AppendAllText("calls.txt", name + Environment.NewLine);
        Console.WriteLine("fixture " + name);
        Console.Error.WriteLine("OpenCL API: fixture BEGIN");
        Console.Error.Flush();
        if (stage == "profile") {
            bool needsHostSeed = mode == "rng-fail" || mode == "rng-timeout" || mode == "combined-fail";
            if (needsHostSeed != (Array.IndexOf(args, "--opencl-host-seed") >= 0)) return 3;
            if (mode == "pair-fail" && Value(args, "--opencl-inverse") != "single") return 3;
        }
        if (mode == "rng-timeout" && stage == "rng") Thread.Sleep(60000);
        if (mode == "smoke-fail" && stage == "smoke") return 2;
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
        $output = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $dir "test-opencl.ps1") -TimeoutSeconds 30 2>&1
        if ($LASTEXITCODE -ne $case.Exit) { throw "$($case.Mode) exit mismatch: $LASTEXITCODE`n$($output -join "`n")" }
        $calls = (Get-Content -LiteralPath (Join-Path $dir "calls.txt")) -join ","
        if ($calls -ne $case.Calls) { throw "$($case.Mode) stage mismatch: $calls" }
        $reports = @(Get-ChildItem -LiteralPath $dir -Recurse -Filter summary.txt)
        if ($reports.Count -ne 1) { throw "$($case.Mode) missing summary" }
        $report = Get-Content -LiteralPath $reports[0].FullName -Raw
        foreach ($expected in @($case.Text, "OpenCL API: fixture BEGIN", "Send summary.txt")) {
            if (-not $report.Contains($expected)) { throw "$($case.Mode) missing '$expected'`n$report" }
        }
        if ($case.Exit -ne 0 -and $report.Contains("Optional search command")) { throw "Failed test suggested a search" }
        if (Get-ChildItem -LiteralPath $dir -Recurse -Filter *.jsonl) { throw "Diagnostic wrote wallet output" }
        Write-Host "Launcher fixture PASS: $($case.Mode)"
    }
} finally {
    $env:TRON_LAUNCHER_FIXTURE = $previousMode
}
# Only remove this test's newly allocated directory after all checks pass.
Remove-Item -LiteralPath $root -Recurse -Force

# Exercise the packaged Vulkan benchmark orchestration without a GPU.
$ErrorActionPreference = "Stop"
$root = Join-Path ([IO.Path]::GetTempPath()) ("tron-vulkan-launcher-test-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $root | Out-Null
$fixture = Join-Path $root "fixture.exe"
Add-Type -OutputAssembly $fixture -OutputType ConsoleApplication -TypeDefinition @'
using System;
using System.IO;
public class Fixture {
    static string Value(string[] args, string name) {
        int i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : "";
    }
    public static int Main(string[] args) {
        bool test = Array.IndexOf(args, "test-vulkan") >= 0;
        bool failTest = File.Exists("fixture-fail-test");
        bool failProfile = File.Exists("fixture-fail-profile");
        bool openclWinner = File.Exists("fixture-opencl-winner");
        string batch = Value(args, "--vulkan-curve-batch");
        string affine = Value(args, "--vulkan-affine-batch");
        string name = test ? "test" : batch == "" ? "opencl" :
            batch == "1" ? "vulkan-baseline" :
            affine == "4" ? "vulkan-curve4-affine4" : "vulkan-winner";
        File.AppendAllText("calls.txt", name + Environment.NewLine);
        if (test) {
            if (failTest) return 2;
            Console.WriteLine("Vulkan resident repeated dispatch + CPU verification PASS (no wallets)");
            return 0;
        }
        if (Value(args, "--words") != "words.txt" || Value(args, "--bench-seconds") != "1" ||
            Array.IndexOf(args, "--no-config") < 0) return 3;
        if (failProfile && batch == "4") return 4;
        if (batch == "") Console.WriteLine("wall 1.000 s, wall speed " +
                                             (openclWinner ? "40.00" : "10.00") + " M/s");
        else Console.WriteLine("Vulkan batch " + batch + " / affine " + affine +
                               " / fixture GPU " +
                               (batch == "4" && affine == "8" ? "30.00" : "20.00") + " M/s");
        return 0;
    }
}
'@
try {
    foreach ($case in @(
        @{ Mode = "success"; Exit = 0; UpdateConfig = $true; Calls = "test,opencl,vulkan-baseline,vulkan-curve4-affine4,vulkan-winner,vulkan-winner,opencl" },
        @{ Mode = "opencl-winner"; Exit = 0; UpdateConfig = $true; Calls = "test,opencl,vulkan-baseline,vulkan-curve4-affine4,vulkan-winner,vulkan-winner,opencl" },
        @{ Mode = "fail-test"; Exit = 1; Calls = "test" },
        @{ Mode = "fail-profile"; Exit = 1; Calls = "test,opencl,vulkan-baseline,vulkan-curve4-affine4,vulkan-winner,vulkan-winner,opencl" }
    )) {
        $dir = Join-Path $root ("test " + $case.Mode)
        New-Item -ItemType Directory -Path $dir | Out-Null
        Copy-Item -LiteralPath $fixture -Destination (Join-Path $dir "tron_vanity_generator.exe")
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot "../bench-vulkan.ps1") -Destination $dir
        Set-Content -LiteralPath (Join-Path $dir "words.txt") -Value "energy" -Encoding ASCII
        if ($case.UpdateConfig) {
            Set-Content -LiteralPath (Join-Path $dir "tron-vanity.conf") -Value @(
                "backend=opencl", "gpu-resident=true", "gpu-group-size=64",
                "opencl-pipeline=staged", "opencl-inverse=pair",
                "vulkan-curve-batch=1", "vulkan-affine-batch=4"
            ) -Encoding ASCII
        }
        if ($case.Mode -eq "fail-test") {
            New-Item -ItemType File -Path (Join-Path $dir "fixture-fail-test") | Out-Null
        } elseif ($case.Mode -eq "fail-profile") {
            New-Item -ItemType File -Path (Join-Path $dir "fixture-fail-profile") | Out-Null
        } elseif ($case.Mode -eq "opencl-winner") {
            New-Item -ItemType File -Path (Join-Path $dir "fixture-opencl-winner") | Out-Null
        }
        $launcherArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $dir "bench-vulkan.ps1"), "-Seconds", "1")
        if ($case.UpdateConfig) { $launcherArgs += "-UpdateConfig" }
        $output = & powershell.exe @launcherArgs 2>&1
        if ($LASTEXITCODE -ne $case.Exit) {
            throw "$($case.Mode) exit mismatch: $LASTEXITCODE`n$($output -join "`n")"
        }
        $calls = (Get-Content -LiteralPath (Join-Path $dir "calls.txt")) -join ","
        if ($calls -ne $case.Calls) { throw "$($case.Mode) call order mismatch: $calls" }
        $reports = @(Get-ChildItem -LiteralPath $dir -Recurse -Filter summary.txt)
        if ($reports.Count -ne 1) { throw "$($case.Mode) missing summary" }
        $report = Get-Content -LiteralPath $reports[0].FullName -Raw
        if ($case.Mode -eq "fail-test") {
            if (-not $report.Contains("Vulkan correctness failed")) { throw "Missing fail-closed report" }
        } else {
            $csv = Import-Csv -LiteralPath (Join-Path $reports[0].Directory.FullName "benchmark.csv")
            if ($csv.Count -ne 7) { throw "$($case.Mode) expected seven benchmark rows" }
            $vulkan4 = @($csv | Where-Object { $_.Test -eq "04-vulkan-curve4-affine4-A" })[0]
            $winner = @($csv | Where-Object { $_.Test -eq "05-vulkan-winner-A" })[0]
            if ($case.Mode -eq "success") {
                if ($vulkan4.Result -ne "PASS" -or $vulkan4.MKeysPerSecond -ne "20" -or
                    $winner.Result -ne "PASS" -or $winner.MKeysPerSecond -ne "30") {
                    throw "Vulkan resident wall rates were not parsed"
                }
            } elseif ($winner.Result -notlike "FAIL*") { throw "Profile failure was not recorded" }
        }
        if ($case.UpdateConfig) {
            $configLines = @(Get-Content -LiteralPath (Join-Path $dir "tron-vanity.conf")) |
                ForEach-Object { $_.TrimStart([char]0xFEFF) }
            if ($case.Mode -eq "opencl-winner") {
                if (-not ($configLines -contains "backend=opencl") -or
                    -not ($configLines -contains "gpu-resident=true") -or
                    -not ($configLines -contains "opencl-pipeline=staged") -or
                    @($configLines | Where-Object { $_ -like "vulkan-*" }).Count) {
                    throw "Successful OpenCL winner did not replace stale Vulkan config: $($configLines -join '|')"
                }
            } elseif (-not ($configLines -contains "backend=vulkan") -or
                      -not ($configLines -contains "vulkan-curve-batch=4") -or
                      -not ($configLines -contains "vulkan-affine-batch=8") -or
                      -not ($configLines -contains "vulkan-batch-keys=131072") -or
                      -not ($configLines -contains "vulkan-resident-group=16") -or
                      @($configLines | Where-Object { $_ -like "opencl-*" }).Count) {
                throw "Successful Vulkan winner did not replace stale OpenCL config: $($configLines -join '|')"
            }
            if (-not (Get-ChildItem -LiteralPath $dir -Filter "tron-vanity.conf.bak-*")) {
                throw "Config backup was not created"
            }
        }
        if (Get-ChildItem -LiteralPath $dir -Recurse -Filter *.jsonl) {
            throw "$($case.Mode) wrote wallet output"
        }
    }
    Write-Host "Vulkan benchmark launcher PASS (success and failure paths)"
} finally {
    if (Test-Path -LiteralPath $root) { [IO.Directory]::Delete($root, $true) }
}
# The final fixture intentionally exits 1. Do not leak that LASTEXITCODE into
# the GitHub Actions PowerShell host after all assertions have passed.
exit 0

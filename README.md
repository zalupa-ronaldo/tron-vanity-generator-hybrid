# TRON Vanity Generator — CPU + OpenCL + CUDA + Metal + experimental Vulkan

[![Windows release](https://img.shields.io/github/v/release/zalupa-ronaldo/tron-vanity-generator-hybrid?display_name=tag)](https://github.com/zalupa-ronaldo/tron-vanity-generator-hybrid/releases)
[![Build](https://github.com/zalupa-ronaldo/tron-vanity-generator-hybrid/actions/workflows/release.yml/badge.svg)](https://github.com/zalupa-ronaldo/tron-vanity-generator-hybrid/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Hybrid TRON vanity-address generator with OpenCL on Windows/Linux/macOS, native CUDA on
Windows/Linux, and native Metal on Apple Silicon. The Windows release embeds
CUDA PTX: NVIDIA users need a compatible driver, without installing the CUDA
Toolkit. AMD GPUs use OpenCL.

The GPU architecture is based on and attributes
[hlzzhqb/tron-vanity-generator](https://github.com/hlzzhqb/tron-vanity-generator)
(MIT). The project also bundles Bitcoin Core `libsecp256k1` as an MIT
submodule; see `NOTICE` and `third_party/secp256k1/COPYING`.

The Metal resident point walk and per-lane batch inversion were inspired by
[mrtozner/tron-vanity-metal](https://github.com/mrtozner/tron-vanity-metal)
(MIT). Our kernel keeps GPU-generated starting scalars and checks the full
dictionary against each Base58 address; benchmark rates from the two projects
therefore measure different workloads and hardware.

## What it searches

`--words` supplies a Base58-compatible dictionary. Every address is scanned
after its initial `T`; lowercase and uppercase variants are matched unless
`--case-sensitive` is used. The OpenCL/CUDA kernel uses a flattened Aho–Corasick
automaton, so CPU and GPU use the same matching semantics.

`words.example.txt` is a curated starter dictionary with 350+ entries across
TRON/energy, payments, names, technology, gaming, space and memorable phrases.
Copy it to `words.txt` and add your own Base58-compatible tokens. The matcher
accepts words anywhere in the address, including the final characters; longer
matches are especially rare and useful.

## Windows: run in one click

Download and extract the [latest Windows ZIP](https://github.com/zalupa-ronaldo/tron-vanity-generator-hybrid/releases).
The ZIP contains `tron_vanity_generator.exe`, `tron-vanity.conf`,
`words.txt` and `bench.cmd`. Replace `words.txt` with your own dictionary if
needed. Double-click **`tron_vanity_generator.exe`**: it reads the adjacent
config and searches on the RX 9070 XT until Ctrl+C. The bundled config uses
the validated staged OpenCL settings (GPU only, paired inversion, affine
batch 4, group 64); it does not enable unmeasured experiments. If the selected
GPU is unavailable, it fails instead of silently running on CPU.

From a terminal in the extracted folder, the short commands are:

```text
tron_vanity_generator.exe           search using tron-vanity.conf
tron_vanity_generator.exe devices   list available hardware
tron_vanity_generator.exe test      verify GPU addresses and private scalars
bench.cmd                           bounded full configuration benchmark
```

`bench.cmd` writes `summary.txt` and `benchmark.csv` in a new
`opencl-diagnostic-*` folder. It never creates wallet files. Search results,
by contrast, contain **unencrypted private keys** in `results`; protect that
folder, never upload it, and move funds only after independently verifying an
address. `tron_vanity_generator.exe bench` also runs an in-process OpenCL
tuning matrix and writes a no-wallet JSON report, but `bench.cmd` is preferred
on Windows because every compiler attempt is separately time-bounded.

See the measured [M4 Metal profile](docs/apple-m4-metal-hardware-profile.md),
[RX 9070 XT OpenCL profile](docs/rx9070xt-opencl-research.md), and
[engineering handoff](docs/next-agent-handoff.md). There is an optional native
Vulkan compute probe for source builds (`-DTRON_ENABLE_VULKAN=ON`, then
`tron_vanity_generator --vulkan-test`). It compiles a small SPIR-V shader,
dispatches it and verifies 256 results against the CPU. This is an API and
compiler check only: **it does not generate wallets or search addresses**.
`test-vulkan` separately exercises a native secp256k1 point walk from a
single test base, then Keccak-256, double SHA-256, Base58Check and dictionary
matching. It checks public points, full 34-character TRON addresses, match
IDs and an atomic bounded result ring against CPU references at workgroup and
offset-window boundaries. It requires Vulkan `shaderInt64`.
The regular Windows ZIP does not include Vulkan. Opt-in Vulkan builds now
also provide an **experimental** `--backend vulkan` full-address wallet backend:
OS CSPRNG chooses a base scalar, GPU computes all address stages and dictionary
matches, and CPU independently verifies every reported key/address/match.
It fails closed on a driver timeout or result-ring overflow. The reusable
backend passes software-Vulkan tests but is **not yet validated on RX 9070 XT**;
do not use it for funds until `test-vulkan` passes on the actual card and its
wallet output is independently checked. It is not expected to beat staged
OpenCL yet because the Vulkan curve shader inverts each point separately.
An opt-in `vulkan-stage-test-windows-x64` executable is also saved as a
short-lived artifact of successful GitHub Actions builds; on an RX 9070 XT it
can run `tron_vanity_generator.exe test-vulkan` without writing wallets.

For a source build on Windows with the [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
installed, use `powershell -ExecutionPolicy Bypass -File .\build.ps1 -EnableVulkan`.
The build runs the Vulkan stage and repeated-dispatch tests locally. The SDK
is a build dependency, not needed by current regular Windows ZIP users.
The installed Vulkan runtime/driver is still required on the target machine.
For a bounded, no-wallet A/B test from the optional executable, use:

```bat
tron_vanity_generator.exe --no-config test-vulkan
tron_vanity_generator.exe --no-config --backend vulkan --words words.txt --bench --bench-seconds 5
```

Use `--no-config` because the bundled default config intentionally selects
OpenCL and includes OpenCL-only options. `--backend vulkan` never silently
falls back to CPU: software Vulkan devices are rejected for normal runs.
For a deliberate no-wallet software-driver benchmark in a development
environment, set `TRON_VULKAN_ALLOW_SOFTWARE=1`. Do not compare that CI rate
with RX hardware.

## Build on Windows

Install Visual Studio Build Tools with the C++ workload, CMake, Ninja and Git.
The GPU path only needs the OpenCL runtime shipped with the AMD or NVIDIA
driver.

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1
.\build\tron_vanity_generator.exe --selftest
.\build\tron_vanity_generator.exe --list
```

## Advanced: run hybrid mode

```powershell
.\build\tron_vanity_generator.exe `
  --seconds 259200 `
  --threads 16 `
  --words ..\tron-vanity\words.txt `
  --out results
```

The console progress line shows elapsed percentage, CPU threads and keys/s,
GPU keys/s, total throughput and match count. `--seconds 0` runs until Ctrl+C.

The old fine-grained flags remain available for controlled experiments, but
normal operation only needs the config and the short commands above. For a
manual CPU+GPU session, use:

```powershell
.\build\tron_vanity_generator.exe --no-config --backend auto --words words.txt --seconds 60
```

If OpenCL is unavailable, `--backend opencl` falls back to CPU. `auto` uses
CPU plus every available OpenCL GPU on Windows/Linux. On macOS, OpenCL is
explicitly selected; the existing Metal/CPU defaults are unchanged. CUDA is selected with
`--backend cuda`; it does not run alongside OpenCL on the same NVIDIA GPU.

Discrete GPUs default to a larger `2^20` GPU batch to reduce command-queue
gaps. Integrated GPUs stay at `2^16` to keep desktop responsiveness. Use
`--gpu-batch` to override this with a power-of-two value from 1024 to 1048576.

## GPU-resident mode (experimental)

`--gpu-resident` keeps the dictionary, address checking and a device result ring
on the GPU. On OpenCL/CUDA, a lightweight GPU RNG writes one 32-byte scalar per
chunk. The host reads that scalar, expands its public point once with
libsecp256k1, uploads 64 bytes, and the GPU scans the consecutive range. This
96-byte round trip and one CPU point multiplication are amortized over the
whole chunk; after that, the CPU receives only complete matches for independent
secp256k1/address verification and local JSONL writing. Keeping the full
256-bit fixed-base multiplication out of OpenCL reduces compiler load, but
older Windows RDNA4 builds timed out during monolithic compilation. Current
staged builds passed all isolated and combined self-tests on the user's
gfx1201/driver 3665.0; see the RX research note. OpenCL dispatch
size adapts toward the requested chunk time; this is not a hard WDDM timeout guarantee:

```powershell
.\build\tron_vanity_generator.exe --backend opencl --gpu-resident `
  --gpu-rng chacha12 --gpu-buffer-mb 128 --gpu-chunk-ms 32 --gpu-poll-ms 50 `
  --seconds 60
```

`chacha12` is the production resident RNG. `philox` and `aes-ctr` are available
for cross-vendor benchmarking; all three use a per-run OS-generated seed and
rejection sampling against the secp256k1 order. If the device result ring
overflows, the run stops rather than silently dropping a wallet result.

The Metal resident backend is available on Apple Silicon with macOS 26+ and
Xcode 26.6's Metal Toolchain:

```bash
sudo xcode-select --switch /Applications/Xcode-26.6.0.app/Contents/Developer
./build-macos.sh
./build-mac-metal/tron_vanity_generator --backend metal --gpu-resident \
  --gpu-rng chacha12 --gpu-buffer-mb 128 --gpu-chunk-ms 32 \
  --seconds 60
```

The MSL kernel is compiled and linked as a Metal library during the macOS
build. The current release keeps Metal opt-in; `--backend auto` will select it
after the cross-validation acceptance run.

Metal walks a compile-time batch of consecutive curve points per lane and uses
one modular inverse for the whole batch. The conservative default remains 32,
but the batch can be any power of two through 1,024 as long as
`group-size * keys-per-lane <= 65536`. A balanced, same-temperature profile on
a base 10-GPU-core M4 selected `--gpu-group-size 128
--metal-keys-per-lane 512`: 34.51 M keys/s versus 31.87 M keys/s for the old
256-by-32 configuration (about 8.3% faster). The production worker subsequently
held 33.3-34.1 M keys/s while matching a 1,665-word dictionary.

The large M4 batch is intentional. Xcode Metal System Trace reports 85,168 B
of compiler spill per thread at 512 keys/lane, but that state is amortized over
512 keys and unified-memory bandwidth remains sufficient. Smaller batches lose
more to secp256k1 inversion than they save in spill traffic. A separate run
independently verified 121 generated address/key pairs at 128-by-512, with no
duplicate keys. Rates from upstream prefix-only search are not directly
comparable to full Base58 dictionary matching.

The Metal path also uses a named-lane Keccak-f[1600] permutation. It keeps the
25 lanes in scalar variables and performs Rho/Pi as one in-place cycle instead
of dynamically indexing three temporary arrays. A same-temperature A/B on the
base M4 measured 27.8 M keys/s versus 23.1 M keys/s for the indexed reference
permutation (about 20% faster); cold short runs reached 34.3 M keys/s. Startup
compares both permutations over 1,024 deterministic blocks.

Already-saved dictionary words are removed from the live DFA output tables
between OpenCL, CUDA and Metal dispatches. This preserves the existing
one-result-per-word behavior without sending millions of duplicate short-word
records back to the CPU. The Metal grid is scaled inversely with keys per lane,
keeping command buffers near the same key count as the batch changes; the tuned
M4 command is about 0.76 seconds instead of growing into a multi-second
dispatch.

```bash
./build-mac-metal/tron_vanity_generator --bench-resident --backend metal \
  --words words.example.txt --bench-seconds 2 --gpu-chunk-ms 100 \
  --gpu-group-size 128 --metal-keys-per-lane 512
```

For stage-by-stage diagnosis, use `--metal-profile-stages`; it compiles
separate EC, Keccak, SHA-256, Base58 and dictionary variants and writes no
wallet file. `--metal-profile-indexed-keccak` selects the old Keccak only for
an A/B profile. `--metal-profile-max-threads N` exposes the Metal compiler's
occupancy hint for controlled experiments; it is not enabled in production
because the M4 A/B stayed within run-to-run noise.

The deterministic hardware profiler measures arithmetic, both field
representations, SIMD shuffles, thread-private spill cliffs, threadgroup
capacity and unified-memory access without using wallet data:

```bash
./build-mac-metal/tron_vanity_generator --metal-hw-profile \
  --bench-seconds 0.25 \
  --metal-hw-json benchmarks/apple-m4-metal-hardware-2026-09-21.json
```

Use `--metal-hw-case field|private|memory|threadgroup|group` to select one
category. The full M4 methodology, Xcode spill measurements and conclusions
are in [docs/apple-m4-metal-hardware-profile.md](docs/apple-m4-metal-hardware-profile.md).

Resident work-group size can be tuned per GPU without rebuilding:

```powershell
.\tron_vanity_generator.exe --backend opencl --gpu-resident `
  --gpu-group-size 256 --gpu-rng philox --seconds 60
```

Supported values are `64`, `128` and `256`; `256` is the default. If a driver
reports a work-group launch limit, retry with `128` or `64`. This does not
change the OpenCL compilation path or fix a compiler hang.

Resident startup prints the lightweight RNG/scan compilation, 32-bit offset
table upload, ring allocation and GPU-CSPRNG/CPU-expansion validation stages
separately. The first launch can spend time in the vendor compiler while it
populates the driver cache; this is CPU-side initialization and therefore does
not show as GPU utilization. The resident build also excludes unrelated RNG
implementations and the legacy/profile/test entry points from each JIT. Every
returned private key/address/dictionary match is recomputed on the CPU before
it can be written. If initialization is still blocked by a vendor driver, the
regular OpenCL path remains available:

```powershell
.\tron_vanity_generator.exe --backend opencl --ec-window 8 `
  --gpu-batch 1048576 --seconds 60
```

### OpenCL startup and optimization diagnostics

The resident path now defaults to `--opencl-compiler compact`: heavy EC
functions are outlined and the inversion squaring loops are not unrolled.
The arithmetic and rejection sampling are unchanged. This limits compiler
code expansion but did not solve the reported RX 9070 XT startup problem.
**v1.6.1 on gfx1201, driver 3665.0 (PAL,LC):** tiny-kernel and RNG-only checks
passed in 1.1/0.7 seconds, while both compact scan-only inversion variants hit
the 30-second limit inside `clBuildProgram`. This isolates the wait to building
the scan program, not context creation, RNG execution or GPU search execution.
It does not identify a specific faulty compiler pass or prove an infinite hang.
`--opencl-compiler default`
retains the inlining-oriented alternative for comparison.

**v1.7.0 on gfx1201, driver 3665.0 (PAL,LC):** the curve, both affine variants,
and dictionary programs built in under three seconds each. The combined address
program (Keccak/SHA256d/Base58) exceeded the 30-second build limit. No scan ran.

`--opencl-pipeline staged` now builds **six separate programs**: curve, affine,
Keccak, SHA256d checksum, Base58Check encoding, and dictionary matching.
The three address operations use independent programs; compact mode outlines
their heavy functions and limits loop unrolling. A shared context and in-order
queue carry the GPU buffers between stages without CPU readback. Unrelated
implementations are removed from each program by preprocessing. This addresses
the observed compiler bottleneck. **v1.7.1 on RX 9070 XT / gfx1201, driver
3665.0 (PAL,LC):** all staged builds, both inversion self-tests, GPU RNG and
the five-second profile passed. With a 358-word dictionary and paired inversion,
the user-reported profile measured 66.870 M keys/s wall throughput (334,364,672
keys in 5.000 seconds); the driver reported 93.099 M/s for GPU kernel time only.
This is a profile result, not a measurement of wallet-output throughput.

Normal CLI search keeps `monolithic` as the pipeline default. Selecting `staged`
requires `--backend opencl` and implies `--gpu-resident`. The staged path uses
34 MiB of additional scratch for public points, hash payloads, checksums and
addresses, capped at 65,536 work-items / 131,072 keys per chunk. It adds five
scan dispatches and GPU-memory traffic, so its speed must be measured on the
target device. GPU event timings sum all six stage events; compare wall
throughput, not just the last kernel.
`--opencl-pipeline staged` selects GPU-resident mode even when `--gpu-resident`
is omitted. To compare with ordinary OpenCL search, omit both options:

```bat
tron_vanity_generator.exe --backend opencl --words words.txt --seconds 60
```

The two user-reported runs near 68.8 M/s both selected `staged` and therefore
used the same resident backend. A separate ordinary OpenCL run was reported at
about 55 M/s on the same RX 9070 XT, making the v1.7.1 resident rate about 25%
higher.
In the comparable v1.7.2 five-second profile, the user reported 75.474 M/s
(377,487,360 keys), a 12.9% increase over v1.7.1's 66.870 M/s profile.

The staged resident path now reuses each random base point for up to 4,194,304
consecutive offsets. The curve and match kernels already accept an offset base;
the host advances it after each verified chunk and rotates the random base
before the 32-bit offset range can wrap. This reduces GPU RNG dispatches, GPU
readback and CPU public-point expansion. The reported v1.7.1 profile spent
0.652 of 5.000 seconds in base preparation; v1.7.2 spent 0.029 seconds on
the same RX 9070 XT and reported 90 newly generated bases. The profile reports
new base count so the rate of rotation can be checked directly.
Keys derived from one base are consecutive rather than independent random
samples. This was already true within each resident chunk; reuse extends that
relationship across chunks. Keep all generated private keys secret: disclosure
of one key together with its relative offset can reveal other keys from that
base window.

The next staged OpenCL revision reduces work in all six kernels: the curve
stage uses three 8-bit offset windows within the bounded 4M-key base window;
the paired affine stage handles four points per inversion; Keccak packs its
64-byte input eight bytes at a time; the fixed-size checksum wrapper is
specialized for its 21- and 32-byte inputs; Base58 divides by 58² to emit two
digits per pass; and the dictionary stage derives a private scalar only after
finding a match. The old two-point affine path remains selectable for an A/B
test with `--opencl-affine-batch 2` (default `4` for staged paired inversion).
An experimental batch of `8` shares one inversion across eight points while
reloading full points only as they are converted. On the local M4 OpenCL
profile, two alternating three-second samples gave 11.399/11.619 M/s for
batch 4 and 13.304/13.617 M/s for batch 8; the affine stage itself fell from
about 0.765 to 0.455 ns/key. This is a candidate, not an RX 9070 XT result.
The batch-8 self-test compares generated addresses and private scalars with
the CPU, including a base-window transition.
The curve stage also has experimental `--opencl-curve-batch 4|8` options
(default `2`). Each work-item computes one offset point, then walks three or
seven adjacent points by adding the generator, amortizing offset-table work.
On M4 OpenCL with affine batch 8, two alternating three-second profiles put
curve batch 2 at 13.281/13.742 M/s and about 0.51 ns/key for the curve stage;
curve batch 4 at 14.417/14.579 M/s and 0.36–0.37 ns/key; curve batch 8 was
14.338/15.130 M/s and 0.38 ns/key. These are not RX 9070 XT measurements.
The curve-batch self-tests verify CPU-equivalent results across base-window
transitions.
The implementation changes were benchmarked with the same five-second wall
profile and six GPU stage timings on the RX 9070 XT:
The user-reported v1.8.0 run on RX 9070 XT / gfx1201 passed all 13 checks
and measured 105.082 M/s wall throughput (525,467,648 keys / 5.001 s),
39.2% above the comparable v1.7.2 profile. The GPU stage totals were curve
0.704 s, affine 1.779 s, Keccak 0.441 s, checksum 0.385 s, Base58 0.427 s,
and match 0.222 s. They processed more keys than v1.7.2, so compare time
per key rather than the unnormalized totals. These are still profile numbers,
not wallet-output search speed.
In a later full diagnostic run on the same reported RX 9070 XT / driver
3665.0, all 16 checks passed. Paired affine batch 2/4/8 measured 83.848,
104.684 and 101.804 M/s wall throughput respectively over five-second
profiles; batch 4 was again the best of those three. Its kernel-only rate
was 132.786 M/s, which is not the user-visible wallet search rate.

```bat
tron_vanity_generator.exe --backend opencl --opencl-pipeline staged --opencl-inverse pair --opencl-affine-batch 4 --gpu-group-size 64 --opencl-profile --words words.txt --bench-seconds 5
tron_vanity_generator.exe --backend opencl --opencl-pipeline staged --opencl-inverse pair --opencl-affine-batch 2 --gpu-group-size 64 --opencl-profile --words words.txt --bench-seconds 5
tron_vanity_generator.exe --backend opencl --opencl-pipeline staged --opencl-inverse pair --opencl-affine-batch 8 --gpu-group-size 64 --opencl-profile --words words.txt --bench-seconds 5
```

For per-stage A/B tests, `--opencl-opt-mask N` accepts a sum of bits:
curve `1`, affine `2`, Keccak `4`, checksum `8`, Base58 `16`, match `32`.
`0` uses the v1.7.2 staged kernels; `63` enables all v1.8.0 optimizations
(default). The mask affects only staged resident OpenCL. Use
`test-opencl.cmd -CompareStages -TimeoutSeconds 120` after the normal
diagnostic to run a bounded, no-wallet five-second profile of baseline and
each single optimization. Its `summary.txt` keeps the speed and six stage
times (also normalized as ns/key) for every variant; full build logs stay in
separate files. A single-bit result is not necessarily additive with the
others; judge the production default `63` by its measured wall speed.
For a bounded, no-wallet comparison of all three affine batch sizes on
Windows, use `test-opencl.cmd -CompareAffineBatches -TimeoutSeconds 120`.
The launcher runs its normal self-tests first and only compares batches if
paired inversion passes. Judge both wall M/s and affine ns/key; compilation
time, register pressure, and the best batch can differ by GPU and driver.
Newer profiles also print `host timing`: enqueue, finish wait, event query,
metadata read, record handling, and metadata update. These are wall-time
subdivisions, not GPU execution time, and help distinguish driver/transfer
overhead from the six kernel timings. A loaded GPU can make the finish wait
much larger than the sum of its own kernel event times.
Use `test-opencl.cmd -CompareCurveBatches -TimeoutSeconds 120` for the same
bounded comparison of curve batch sizes 2, 4, and 8. The two comparison
switches can be combined, but test them one at a time when isolating effects.
The optional `--opencl-sha-ring` uses a 16-word circular message schedule
instead of 64 words in the staged checksum kernel. It passed the GPU/CPU
address and key self-test but was slower on the concurrently loaded M4
(roughly 2.0 versus 1.5–1.6 ns/key for checksum); it is off by default.
`test-opencl.cmd -CompareShaRing -TimeoutSeconds 120` runs a bounded,
no-wallet comparison on the target GPU before considering it for production.
`test-opencl.cmd -CompareGroupSizes -TimeoutSeconds 120` compares local group
sizes 64, 128, and 256 with the same staged pipeline and dictionary. It
checks GPU/CPU address and private-scalar agreement at each new group size
before profiling it; an unsupported group stops the comparison without
starting a real search. Compare wall M/s and per-stage ns/key rather than
assuming a larger group is faster on every GPU.
The optional `--opencl-async-meta-read` queues the small result-metadata
read after the last staged kernel, before the existing `clFinish`, instead
of issuing a separate blocking read afterward. The same in-order queue and
finish still guard consumption of the result. It is off by default until the
target GPU is measured. Run
`test-opencl.cmd -CompareMetaRead -TimeoutSeconds 120` for a bounded,
no-wallet scan self-test followed by five-second blocking/queued profiles.
The `metadata read API` time measures only the enqueue call in queued mode;
its completion is part of `finish wait`. Judge the wall M/s and total host
overhead, not that API number alone.

`--opencl-inverse pair` uses one field inversion for two Jacobian points in the
monolithic path; staged `--opencl-affine-batch 4` uses one for four. It avoids
work-group barriers and large point arrays; infinity is masked out of the
product so it cannot corrupt its neighbor. `single` remains the conservative
default until target-GPU measurements establish the register/throughput tradeoff.
These options affect resident OpenCL only, not legacy OpenCL, Metal or CUDA.

On Windows, extract the release to a new folder (preserve your own `words.txt`)
and double-click **`test-opencl.cmd`**, or run it from CMD. It now runs separate,
bounded checks and defaults to the **staged** pipeline: tiny OpenCL kernel,
RNG only, build-only checks for each stage (both affine inversion variants),
then full scan self-tests. Build checks continue after a failing stage so the
report can localize the problematic compilation unit. A build PASS is not an
execution/correctness PASS. The full scan tests only run if all required units
and at least one affine variant built successfully. After that, GPU-RNG plus
staged-scan is tested if RNG passed. If no scan self-test passes, it stops
without a profile or real-search suggestion. Otherwise it runs a five-second profile using the
validated configuration. A failed GPU RNG/combined check can select the explicit
OS-seeded scan-only workaround; this is not a silent CPU search fallback.

The launcher prints elapsed time and stops only its own child after **30 seconds
per check**. Override with `test-opencl.cmd -TimeoutSeconds 120` if necessary;
a timeout alone cannot distinguish a slow build from a hang. It never writes
wallets or prints private keys, and does not clear caches or change drivers.
Send **`summary.txt`** from the newly created `opencl-diagnostic-*` folder.
The report includes device/driver versions, flags, separate API begin/return
markers and stage results. Normal search has no launcher timeout.
Use `test-opencl.cmd -Inverse single` to prefer single inversion if it passes.
Use `test-opencl.cmd -Pipeline monolithic` only to retest the old pipeline.

`--opencl-host-seed` uses the OS CSPRNG (BCryptGenRandom on Windows) for each
base scalar, with secp256k1 rejection validation, and excludes the GPU RNG
kernel from compilation. Address generation and matching remain on the GPU.
Use only if the scan-only self-test passes; it cannot bypass a blocked scan
compiler. This option requires `--backend opencl` and implies `--gpu-resident`.
The supplied v1.6.1 report already rules out GPU RNG as the cause of that scan
build timeout: the OS-seeded scan timed out too.

Manual CMD commands (one command per line; unlike the launcher, not time-bounded):

```bat
tron_vanity_generator.exe --backend opencl --opencl-diagnose smoke
tron_vanity_generator.exe --backend opencl --opencl-diagnose rng
tron_vanity_generator.exe --backend opencl --opencl-diagnose build-curve
tron_vanity_generator.exe --backend opencl --opencl-diagnose build-affine --opencl-inverse pair
tron_vanity_generator.exe --backend opencl --opencl-diagnose build-keccak
tron_vanity_generator.exe --backend opencl --opencl-diagnose build-checksum
tron_vanity_generator.exe --backend opencl --opencl-diagnose build-base58
tron_vanity_generator.exe --backend opencl --opencl-diagnose build-match
tron_vanity_generator.exe --backend opencl --opencl-pipeline staged --opencl-diagnose scan --opencl-inverse pair
tron_vanity_generator.exe --backend opencl --opencl-pipeline staged --opencl-diagnose full --opencl-inverse pair
tron_vanity_generator.exe --backend opencl --opencl-pipeline staged --opencl-profile --opencl-inverse pair --words words.txt --bench-seconds 5
```

Each staged scan/full self-test independently verifies 1,536 address/key pairs
against libsecp256k1: two consecutive chunks using one base and a third after
forced base rotation. Monolithic self-tests verify 1,024 pairs. The launcher
prints an optional search command for the validated
configuration but never executes it. For a passing staged GPU-RNG/paired test:

```bat
tron_vanity_generator.exe --backend opencl --opencl-pipeline staged --opencl-inverse pair --words words.txt --seconds 60
```

The profile reports full wall throughput, the number of newly generated base
points, base-point preparation, scan/wait
time, remaining host/metadata time, and driver event timestamps, including the
six separate GPU stage totals for the staged path. It exercises
full secp256k1/Keccak/SHA256d/Base58/dictionary math, but excludes CPU match
verification and wallet output. **Compare wall speed, not kernel-only speed**:
event timing excludes queueing and transfers and is driver-reported. Initial
OpenCL work size is 16,384 threads and adapts in work-group multiples, with
growth capped at 2x per dispatch. Printed private-memory bytes are an OpenCL
query, not a count of AMD VGPRs or a measurement of spills.

Local checks passed on Apple M4 OpenCL for monolithic/staged single/pair inversion
and all three RNGs. CPU oracle tests compare both exact pipelines, including
order-boundary/infinity handling, 32-bit offsets, ring overflow and disabled matches. A short,
contended **monolithic** M4 run using 358 words measured 2.66 M/s single versus 4.32 M/s pair,
with paired repeats near 4.0 M/s. The production Metal worker was left running:
these are preliminary wall timings, **not an AMD result or an isolated benchmark**.
Linux CI additionally executes the OpenCL kernels with PoCL on CPU; Windows CI
builds the binary but does not have an AMD GPU. New optimizations still require
measurement on the RX 9070 XT before claiming a speed improvement.

OpenCL objects now have explicit ownership and are released between benchmark
variants (previously contexts/programs/buffers were kept until process exit).
The loader also supports Linux and macOS so the actual OpenCL path can be tested
without substituting Metal. On Linux, install your vendor's OpenCL ICD/runtime;
normal device selection never treats a CPU-only OpenCL ICD as a GPU.

References: [Khronos event profiling](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clGetEventProfilingInfo.html),
[program builds](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clBuildProgram.html),
[explicit context platform selection](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clCreateContext.html),
[kernel resource queries](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clGetKernelWorkGroupInfo.html),
and [AMD occupancy/register tradeoffs](https://gpuopen.com/learn/occupancy-explained/).

### Full benchmark matrix

`--bench` compares every backend available on the current machine. With
`--backend auto` it runs the CPU worker, the legacy OpenCL tuning matrix,
resident OpenCL/CUDA for `chacha12`, `aes-ctr` and `philox`, and resident Metal for
the same three RNGs on Apple Silicon. Explicit `--backend cpu|opencl|cuda|metal`
restricts the matrix to that backend. The legacy OpenCL section also scans
EC window, Montgomery batch size, keys-per-item, match length and GPU batch
size.

On Windows, use `--bench-resident` to skip the legacy tuning matrix and go
straight to the resident OpenCL RNG comparison. For a 9070 XT that stalls at
startup, use `test-opencl.cmd` first instead of compiling all three variants:

```powershell
.\tron_vanity_generator.exe --backend opencl --bench-resident `
  --words words.txt --bench-seconds 15 --gpu-chunk-ms 100 --gpu-poll-ms 1000
```

Use `--bench-seconds` to control the timed duration of each row (default 1
second; the legacy tuning matrix has additional warm-up passes):

```bash
./build-mac-metal/tron_vanity_generator --bench --backend auto \
  --words words.example.txt --bench-seconds 1
```

The benchmark never writes wallet JSONL or private keys. It reports
unavailable backends and continues with the methods that the machine supports.

## Results and security

Runtime output is JSONL with `address`, `words` and `private_key`. Private
keys are written only to the local result file and are never printed. Treat
that file as secret material, do not commit it, and do not upload it to a
public repository. The repository `.gitignore` excludes results, logs, PID
files and generated address tables.

The release archive contains only the executable, a starter dictionary and
the project documentation. It does not contain wallet exports, private keys,
local logs or benchmark output.

## Native CUDA (experimental)

The CUDA backend uses NVIDIA's Driver API and precompiled PTX, sharing the
resident integer secp256k1, Keccak, SHA-256, Base58 and dictionary code with
OpenCL. CUDA performs the RNG and range scan; libsecp256k1 expands one base
point per chunk on the CPU and independently verifies every returned match.
Device buffers, the CUDA context, module and stream are released when the
backend is destroyed. Contexts are rebound when work moves to a worker thread.

Requirements: NVIDIA compute capability 5.2+ and a driver supporting CUDA 12.6
PTX (or newer). AMD RX 9070 XT and Apple GPUs do not support native CUDA.
CUDA failure is reported explicitly, without silently running on the CPU.

Windows CMD commands (each command is one line):

```bat
tron_vanity_generator.exe --list
tron_vanity_generator.exe --backend cuda --gputest
tron_vanity_generator.exe --backend cuda --words words.txt --seconds 60
tron_vanity_generator.exe --backend cuda --bench-resident --words words.txt --bench-seconds 3
```

CUDA always uses the bounded resident pipeline; `--gpu-resident` is optional.
Use `--gpu-group-size 64|128|256`, `--gpu-chunk-ms 8..100`, and
`--gpu-buffer-mb` to tune it. `--gpu-batch`, `--ec-window`, `--keys-per-item`
and `--mont-n` are legacy OpenCL settings. `--gpu-poll-ms` is retained for CLI
compatibility; each resident dispatch already waits for and drains its results,
so an extra idle delay is no longer inserted. `chacha12` is the default RNG;
the other two modes are for benchmarking, not recommended for funded wallets.

`--backend cuda --gputest` checks 1,024 GPU-produced key/address pairs against
libsecp256k1 per RNG, across two chunks, without saving wallets or printing
private keys. Build-time CTest also executes the shared kernel body on CPU
and checks scalar boundaries, address encoding, ring overflow and removed
dictionary outputs. These tests and successful PTX compilation do not establish
GPU throughput or replace validation on an actual NVIDIA device.

To include CUDA when building on Windows:

```powershell
python -m pip install nvidia-cuda-nvrtc-cu12==12.6.85
powershell -ExecutionPolicy Bypass -File .\build.ps1 -EnableCuda
```

To build on Linux (C++ toolchain, CMake, Python and initialized submodules):

```bash
python3 -m pip install nvidia-cuda-nvrtc-cu12==12.6.85
cmake -S . -B build-cuda -DTRON_ENABLE_CUDA=ON
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
./build-cuda/tron_vanity_generator --backend cuda --gputest
```

Only NVIDIA's NVRTC compiler component is needed at build time, even on a
machine without a GPU. It compiles all three RNG variants into embedded PTX;
the shipped executable loads only the NVIDIA driver. For cross-compilation,
generate the header using `scripts/build_cuda_ptx.py` on Windows/Linux and pass
`-DTRON_CUDA_PTX_HEADER=/absolute/path/cuda_ptx.h` with `TRON_ENABLE_CUDA=ON`.
Without this option, a standard build keeps CUDA disabled and reports that in
`--list`.

Implementation references: [NVIDIA NVRTC](https://docs.nvidia.com/cuda/nvrtc/)
and [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/).

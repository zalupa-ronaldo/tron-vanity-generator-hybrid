# RX 9070 XT / gfx1201: measured OpenCL behavior

Date: 2026-09-22/23. This is a redacted record of user-run tests, not a
measurement on the development M4. The structured numbers are in
[`benchmarks/amd-rx9070xt-opencl-2026-09-22.json`](../benchmarks/amd-rx9070xt-opencl-2026-09-22.json).
No wallet file or private scalar is required to reproduce the performance
analysis.

## Fixed workload and correctness

Windows, RX 9070 XT (`gfx1201`), driver `3665.0 (PAL,LC)`, AMD OpenCL 2.0,
OpenCL C 2.0, builds requested with `-cl-std=CL1.2`. The dictionary had 358
words. Each profile lasted five seconds and used the *full* point-walk →
Keccak → SHA-256d → Base58Check → dictionary match workload, not a short
prefix-only counter. Staged compiler mode was `compact`, paired inversion,
local group 64, curve batch 2. Smoke, RNG, each isolated stage build, both
scan modes, and the combined GPU/CPU self-test passed.

The repeated production-style profile measured 105.58 M keys/s wall rate.
This agrees with an earlier 105.15 M keys/s run to within about 0.4%. It does
not prove how fast a funded-wallet search will be if output/verification is
frequent: the profile deliberately excludes CPU match verification and file
writing.
That profile used an 8 MiB result ring. The packaged double-click config
uses 128 MiB to leave more room for bursts of matches; a matched wall-rate
comparison between 8 and 128 MiB has not yet been reported. Do not claim the
128 MiB search runs at 105.58 M/s from this profile alone. `bench.cmd` now
includes 8/128/8 MiB and 16/32/64 ms chunk rows for that measurement.

| Paired affine batch | Wall M keys/s | Affine ns/key | Decision |
| ---: | ---: | ---: | --- |
| 2 | 83.85 | 5.729 | Reject |
| 4 | **104.68** | **3.405** | Keep in RX config |
| 8 | 101.80 | 3.696 | Possible later retest, not default |

The batch-4 versus batch-8 gap is about 2.8% in these five-second runs. It is
plausibly related to occupancy/register pressure, but there is no compiler
VGPR count or hardware occupancy trace here; that explanation remains a
hypothesis. Batch 2 is materially worse (about 20% lower wall throughput).

At 105.58 M/s the six driver-reported stage times were 1.330, 3.383, 0.819,
0.718, 0.808 and 0.416 ns/key for curve, affine, Keccak, checksum, Base58 and
match respectively. Affine is the dominant shader stage. The profile also
reported 0.456 s in the separate blocking metadata-read API across the
five-second run. That API time is *not* a promised removable 0.456 s: queued
read completion can move into `clFinish`, and only a full wall-rate A/B can
establish the net gain. The new queued mode is therefore off by default.

## Hardware facts versus inference

AMD's [GPU specifications](https://rocmdocs.amd.com/en/develop/reference/gpu-specs.html)
identify the 9070 XT as 64 CUs, 16 GiB, `gfx1201`, wave size 32 or 64, and
128 KiB LDS per CU. Those are published architectural properties, not a
measurement of this kernel. AMD's [RDNA 4 ISA guide](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf)
documents the instruction model. AMD's [GPA counter table](https://gpuopen.com/manuals/gpu_performance_api_manual/gpu_performance_api_manual-graphics_counter_tables_gfx12/)
lists wave occupancy and VGPR/LDS/scratch limiters that would help test the
register-pressure hypothesis. We have not captured those counters on this
Windows machine.

## Next controlled experiments

1. Run `bench.cmd` from the extracted Windows release with the *same* 358-word
   `words.txt`, while no game, browser GPU workload or other miner is active.
   Share only `summary.txt` and `benchmark.csv`, never `wallets-*.jsonl`.
2. Compare 64/128/256 local size, curve batch 2/4/8, affine batch 2/4/8,
   16-word SHA schedule, queued metadata read, and cross-combinations.
   Each row gets a bounded GPU/CPU key/address check before timing.
3. Repeat the top two candidates in A/B/A order for at least five seconds per
   row; reject a change unless the difference exceeds run-to-run noise and
   all self-tests pass. Keep reported wall M/s separate from kernel-only ns/key.
4. If AMD's profiling tools are available, capture affine VGPR/scratch
   pressure and wave occupancy for batches 4/8. A larger batch can reduce
   inversion count while increasing register lifetime; the optimum is not
   monotonic.
5. Record driver version and power/clock state with every future RX result.
   Do not present the earlier anecdotal ~55 M/s legacy mode as a matched A/B:
   the exact workload, build and conditions were not recorded together.

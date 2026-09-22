# Apple M4 Metal hardware profile

Date: 2026-09-21. Device: base Apple M4 with 10 GPU cores, reported by Metal
as `applegpu_g16g` / Apple GPU family 9.

This pass measures the actual generator workload and small deterministic Metal
kernels. No wallet or production key data enters the microbenchmarks. Full raw
microbenchmark output is stored in
[`benchmarks/apple-m4-metal-hardware-2026-09-21.json`](../benchmarks/apple-m4-metal-hardware-2026-09-21.json).

## What the public interface exposes

On this machine Metal reports:

- SIMD execution width: 32 threads for every tested pipeline.
- Maximum threads per threadgroup: 1,024.
- Maximum threadgroup memory: 32 KiB.
- Unified memory and an 11.84 GiB recommended working set.
- One counter set, containing only the GPU timestamp counter.

Apple does not publicly specify the M4 GPU register-file size, register-bank
mapping, execution-unit count per core or native instruction latencies. The
analysis therefore separates facts returned by Metal/Xcode from inferences
based on timing. The relevant primary references are Apple's
[threadgroup sizing guide](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes),
[Metal compute optimization talk](https://developer.apple.com/videos/play/tech-talks/10580/),
[GPU occupancy guide](https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy),
[command-buffer GPU timestamps](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime),
and [GPU counter sample buffers](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers).

## Method

The reusable profiler is embedded in the executable and can be run with:

```bash
./build-mac-metal/tron_vanity_generator --metal-hw-profile \
  --bench-seconds 0.25 \
  --metal-hw-json benchmarks/apple-m4-metal-hardware-2026-09-21.json
```

It uses deterministic inputs, writes every result to a Metal buffer, and times
completed command buffers with `GPUStartTime` / `GPUEndTime`. Arithmetic rates
are kernel-update throughput, not claims about one-to-one native instructions.
Optimized AIR LLVM was inspected to confirm that the measured loops and integer
multiplications remain present.

Metal System Trace supplied compiler spill events. The production worker was
temporarily unloaded from `launchd` for each controlled run and restored after
it, preventing a second process from sharing the GPU. Production comparisons
used the same 1,665-word dictionary and the same approximately 26.2 million
keys per command in forward and reverse order.

## Hardware findings

### Field representation

| Kernel | Throughput | Relative |
| --- | ---: | ---: |
| Production secp256k1 10x26 | 1,738 M field multiplications/s | 2.13x |
| Upstream-style 8x32 | 815 M field multiplications/s | 1.00x |

Both outputs were independently recomputed with Python big integers modulo
`2^256 - 2^32 - 977`; all six field rows in the final profile matched. The M4
result strongly favors the existing libsecp256k1-derived 10x26 formula over a
straight port of the upstream 8x32 multiplication. Replacing 10x26 merely to
use fewer limbs would be a regression on this hardware.

### Thread-private arrays and compiler spill

| Logical array | Metal System Trace spill/thread | Throughput |
| ---: | ---: | ---: |
| 128 B (`uint[32]`) | 0 B | 17.86 G updates/s |
| 512 B (`uint[128]`) | 528 B | 17.36 G updates/s |
| 2 KiB (`uint[512]`) | 2,064 B | 1.28 G updates/s |
| 4 KiB (`uint[1024]`) | not separately traced | 0.60 G updates/s |
| 8 KiB (`uint[2048]`) | not separately traced | 0.35 G updates/s |

The generic random-index test has a clear performance cliff between 512 B and
2 KiB per thread. The production 32-key kernel spills 6,464 B/thread; the tuned
512-key kernel spills 85,168 B/thread. The latter is nevertheless faster
because one field inversion is amortized over 512 keys. Its spilled state is
about 166 B per key, while sequential unified-memory kernels sustain roughly
69-83 GB/s. At 34.5 M keys/s, even a conservative read-plus-write estimate is
well below that bandwidth ceiling. This bandwidth comparison is an inference,
not a direct byte counter.

### Threadgroup memory

| Dynamic allocation | Barrier throughput |
| ---: | ---: |
| 1 KiB | 199.35 G/s |
| 8 KiB | 199.58 G/s |
| 16 KiB | 164.57 G/s |
| 24 KiB | 114.69 G/s |
| 32 KiB | 76.82 G/s |

The capacity step begins above 8 KiB. Strides 1 through 32 at a fixed 32 KiB
remain in a narrow 78-84 G/s band, so this test does not expose a material bank
conflict. Capacity/occupancy and barriers dominate. Moving the complete point
batch into 20-32 KiB of threadgroup memory is therefore unlikely to be a free
fix.

### Integer, SIMD and memory behavior

- Four independent 32-bit multiply-add chains reach 436 G updates/s; the
  analogous 64-bit test reaches 112 G updates/s.
- SIMD shuffle work reaches 645 G counted shuffle-updates/s.
- Coalesced read, write and copy measure approximately 83, 75 and 69 GB/s;
  scattered 16-byte reads plus a 16-byte sink measure 29 GB/s.
- The compiler occupancy hint (`maxTotalThreadsPerThreadgroup`) did not change
  production spill and stayed within approximately 1% run-to-run variation.
- Full production commands are 99.8-99.9% GPU busy, so host double-buffering is
  not the current limiter.

## Production batch search

All rows below process the same approximate number of keys per command. Each
configuration was measured in forward and reverse order.

| Threads/group | Keys/lane | Wall rate (mean) | GPU command (mean) | Change vs old |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 32 | 31.87 M/s | 821.65 ms | baseline |
| 256 | 256 | 33.58 M/s | 779.82 ms | +5.4% |
| 128 | 512 | **34.51 M/s** | **759.19 ms** | **+8.3%** |
| 64 | 1024 | 34.10 M/s | 768.10 ms | +7.0% |

The selected M4 configuration is therefore:

```text
--gpu-group-size 128 --metal-keys-per-lane 512 --gpu-chunk-ms 100
```

The grid now scales inversely with keys/lane, keeping command size bounded when
the inversion batch grows. Batch 2,048 was rejected: compiling that private
working set repeatedly interrupted the Metal compiler XPC service. The public
limit remains 1,024 keys/lane, with `group-size * keys-per-lane <= 65536` so the
existing two-byte point offset stays valid.

In a separate correctness run, 121 generated key/address pairs from the chosen
128-by-512 kernel were recomputed with `tronpy`; every address and dictionary
match was valid and all private keys were unique. The temporary file was mode
`0600` and was deleted after verification. No private key appears in this
report or the raw hardware JSON.

## Decision and next optimization

The deployed change is the low-risk batch geometry plus bounded grid sizing.
It raises the live worker to roughly 33.3-34.1 M keys/s on the 1,665-word
dictionary.

The next architectural prototype should preserve a large inversion batch while
making its storage explicitly coalesced. Two candidates deserve measurement:

1. A struct-of-arrays device scratch buffer for point and prefix state, to
   replace opaque compiler spill with predictable lane-coalesced traffic.
2. A SIMD32 cooperative batch inversion that holds one or a small tile of
   points per lane and exchanges field limbs with `simd_shuffle`.

Simply reducing the batch, replacing 10x26 with 8x32, filling all 32 KiB of
threadgroup memory, or adding host-side double buffering is contradicted by the
measurements above.

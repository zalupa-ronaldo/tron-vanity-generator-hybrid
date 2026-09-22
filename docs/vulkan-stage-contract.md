# Native Vulkan pipeline: stage contract and verification gates

The native Vulkan wallet backend is **not yet complete**. Source builds with
`-DTRON_ENABLE_VULKAN=ON` have a generic dispatch probe and a first useful
SPIR-V stage, Keccak-256 from a secp256k1 public key to a 21-byte TRON
payload. Neither test is a wallet search or a throughput benchmark. The
Windows release remains OpenCL/CUDA only.

## Verified interface to preserve

`vulkan/keccak.comp` takes 16 little-endian `uint32_t` words per 64-byte
uncompressed public key (`X||Y`, no `0x04` prefix) at storage binding 0. It
outputs six little-endian words at binding 1: byte `0x41`, the last 20 bytes
of Keccak-256, and three zero padding bytes. A four-byte push constant is the
key count. The test uses 256 deterministic, unfunded secp256k1 test scalars;
the host computes public keys and CPU Keccak reference values and compares
every output word. The test never prints or saves a scalar.

This is deliberately a simple 32-bit buffer ABI. Before performance work,
benchmark the unpack/pack cost and consider an aligned packed layout that
the curve, Keccak, SHA-256 and Base58 stages can share without conversions.
Do not change byte order based only on a single address sample. Add multiple
CPU/GPU vectors covering each input/output word and the final partial word.

## Remaining gates to a selectable `--backend vulkan`

1. Port the OpenCL staged curve and affine math with the same 128-byte point
   wire layout or a documented replacement. Compare every scalar/public key
   with CPU `libsecp256k1`, including the random-base window rollover.
2. Port double SHA-256 and Base58Check, then compare complete 34-character
   TRON addresses with the CPU for each test key. Preserve the `T` prefix and
   final checksum; a payload-only or prefix-only rate is not comparable.
3. Port the dictionary matcher and bounded result ring. Test multiple matches
   per address, wraparound and overflow. Overflow must stop the search, never
   silently discard found keys.
4. Integrate dispatch with the existing `Backend` interface, CLI/config,
   OS CSPRNG base generation, output verification and error handling. Never
   fall back to CPU silently when a strict GPU backend is requested.
5. Add Windows Vulkan SDK CI build, then test on the actual RX 9070 XT. Mesa
   software Vulkan verifies logic/API only. Compare full wall keys/s against
   the same dictionary and GPU workload under staged OpenCL; also measure
   stage ns/key and transfer overhead. Do not recommend Vulkan on performance
   grounds until correctness and matched A/B/A runs pass.

Khronos's [compute guide](https://docs.vulkan.org/guide/latest/compute_shaders.html)
and [shader interface specification](https://docs.vulkan.org/spec/latest/chapters/interfaces.html)
define dispatch/descriptor requirements. If experimenting with `clspv`, use
its [OpenCL-C mapping](https://github.com/google/clspv/blob/main/docs/OpenCLCOnVulkan.md)
and inspect the produced descriptor map; `clvk` is a compatibility route, not
the native backend above.

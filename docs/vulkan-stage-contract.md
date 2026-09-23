# Native Vulkan pipeline: stage contract and verification gates

The native Vulkan wallet backend is **not yet complete**. Source builds with
`-DTRON_ENABLE_VULKAN=ON` have a generic dispatch probe and a first useful
SPIR-V Keccak-256 stage from a secp256k1 public key to a 21-byte TRON
payload, plus SHA-256d from that payload to a checksum-bearing 25-byte
binary address, then Base58Check to full 34-character TRON address text.
Neither test is a wallet search or a throughput benchmark. The
Windows release remains OpenCL/CUDA only.

## Verified interface to preserve

`vulkan/keccak.comp` takes 16 little-endian `uint32_t` words per 64-byte
uncompressed public key (`X||Y`, no `0x04` prefix) at storage binding 0. It
outputs six little-endian words at binding 1: byte `0x41`, the last 20 bytes
of Keccak-256, and three zero padding bytes. A four-byte push constant is the
key count. The test uses 256 deterministic, unfunded secp256k1 test scalars;
the host computes public keys and CPU Keccak reference values and compares
every output word. `vulkan/checksum.comp` then reads binding 1 and writes
seven little-endian words at binding 2: the 25-byte address plus three zero
padding bytes. A compute-to-compute barrier separates the two dispatches;
both intermediate and final buffers are checked. The test never prints or
saves a scalar.
`vulkan/base58.comp` reads binding 2 and writes nine little-endian words at
binding 3 (34 ASCII characters and two zero padding bytes). The test compares
this final text with `tronAddressFromPubXY()` on every vector. It uses the
same two-digit (base 58²) long division as the OpenCL resident kernel.

This is deliberately a simple 32-bit buffer ABI. Before performance work,
benchmark the unpack/pack cost and consider an aligned packed layout that
the curve, Keccak, SHA-256 and Base58 stages can share without conversions.
Do not change byte order based only on a single address sample. Add multiple
CPU/GPU vectors covering each input/output word and the final partial word.

## Remaining gates to a selectable `--backend vulkan`

1. Port the OpenCL staged curve and affine math with the same 128-byte point
   wire layout or a documented replacement. Compare every scalar/public key
   with CPU `libsecp256k1`, including the random-base window rollover.
2. Integrate the verified address stages into a reusable bounded Vulkan
   dispatch path rather than the current one-shot test harness. Preserve the
   `T` prefix and final checksum; a payload-only or prefix-only rate is not
   comparable. Add varying batch sizes, boundary counts and driver-error
   tests before permitting production output.
3. Port the dictionary matcher and bounded result ring. Test multiple matches
   per address, wraparound and overflow. Overflow must stop the search, never
   silently discard found keys.
4. Integrate dispatch with the existing `Backend` interface, CLI/config,
   OS CSPRNG base generation, output verification and error handling. Never
   fall back to CPU silently when a strict GPU backend is requested.
5. Windows Vulkan SDK CI now compiles the optional stages, but runtime has
   only been verified with Mesa software Vulkan on Linux. Test the stage
   executable on the actual RX 9070 XT before enabling any production use.
   Once the full backend exists, compare full wall keys/s against
   the same dictionary and GPU workload under staged OpenCL; also measure
   stage ns/key and transfer overhead. Do not recommend Vulkan on performance
   grounds until correctness and matched A/B/A runs pass.

Khronos's [compute guide](https://docs.vulkan.org/guide/latest/compute_shaders.html)
and [shader interface specification](https://docs.vulkan.org/spec/latest/chapters/interfaces.html)
define dispatch/descriptor requirements. If experimenting with `clspv`, use
its [OpenCL-C mapping](https://github.com/google/clspv/blob/main/docs/OpenCLCOnVulkan.md)
and inspect the produced descriptor map; `clvk` is a compatibility route, not
the native backend above.

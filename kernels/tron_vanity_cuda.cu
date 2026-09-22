// Compile the shared, integer-only resident math as native CUDA C++.
// No OpenCL runtime, source translation at runtime, or CUDA runtime library.
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long long ulong;

__device__ __forceinline__ uint cuda_atomic_add(volatile uint* p, uint v) {
    return atomicAdd(const_cast<uint*>(p), v);
}
#define CUDA_BACKEND 1
#define RESIDENT 1
#define ECW 8
#define ECBITS 32
#define KPI 2
#define MONT_N 1
#define __global
#define __private
#define __constant __device__ __constant__
#define __kernel extern "C" __global__
#define inline __device__ __forceinline__
#define atomic_add cuda_atomic_add
#define mul_hi __umulhi
#define get_global_id(dim) (blockIdx.x * blockDim.x + threadIdx.x)
#include "secp256k1_tron.cl"

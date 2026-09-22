cmake_minimum_required(VERSION 3.20)

# 静态链接 MSVC 运行库，生成的 exe 不依赖 VC++ 运行时可再发行组件，方便复制到别的机器。
# 必须在 project() 之前设策略，且要在 add_subdirectory(secp256k1) 之前，保证两边一致。
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
option(STATIC_RUNTIME "静态链接 C/C++ 运行库（便携 exe）" ON)

project(tron_vanity_generator CXX)

option(TRON_ENABLE_CUDA "Embed native NVIDIA CUDA PTX kernels" OFF)
set(TRON_CUDA_PTX_HEADER "" CACHE FILEPATH "Precompiled cuda_ptx.h for cross-compilation")
if(TRON_ENABLE_CUDA)
  set(CUDA_GEN ${CMAKE_CURRENT_BINARY_DIR}/gen/cuda_ptx.h)
  if(TRON_CUDA_PTX_HEADER)
    configure_file(${TRON_CUDA_PTX_HEADER} ${CUDA_GEN} COPYONLY)
  else()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_custom_command(OUTPUT ${CUDA_GEN}
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_cuda_ptx.py
              --source-dir ${CMAKE_CURRENT_SOURCE_DIR}/kernels --output ${CUDA_GEN}
      DEPENDS scripts/build_cuda_ptx.py kernels/tron_vanity_cuda.cu kernels/secp256k1_tron.cl
      VERBATIM)
  endif()
endif()

if(APPLE)
  enable_language(OBJCXX)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "Minimum macOS for Metal backend")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(STATIC_RUNTIME)
  if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  elseif(MINGW)
    add_link_options(-static -static-libgcc -static-libstdc++)
  elseif(UNIX AND NOT APPLE)
    # GPU driver libraries (OpenCL/CUDA) need the system libc/dynamic loader.
    add_link_options(-static-libgcc -static-libstdc++)
  endif()
endif()

# ---- libsecp256k1 (third_party/secp256k1, git submodule/clone) ----
set(SECP256K1_INSTALL OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_EXHAUSTIVE_TESTS OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_CTIME_TESTS OFF CACHE BOOL "" FORCE)
set(SECP256K1_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SECP256K1_ENABLE_MODULE_ECDH OFF CACHE BOOL "" FORCE)
set(SECP256K1_ENABLE_MODULE_EXTRAKEYS OFF CACHE BOOL "" FORCE)
set(SECP256K1_ENABLE_MODULE_SCHNORRSIG OFF CACHE BOOL "" FORCE)
set(SECP256K1_ENABLE_MODULE_MUSIG OFF CACHE BOOL "" FORCE)
set(SECP256K1_ENABLE_MODULE_ELLSWIFT OFF CACHE BOOL "" FORCE)
set(SECP256K1_ENABLE_MODULE_SILENTPAYMENTS OFF CACHE BOOL "" FORCE)
set(SECP256K1_ENABLE_MODULE_RECOVERY OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/secp256k1 EXCLUDE_FROM_ALL)

# ---- 内核嵌入 ----
set(KERNEL_CL  ${CMAKE_CURRENT_SOURCE_DIR}/kernels/secp256k1_tron.cl)
set(KERNEL_GEN ${CMAKE_CURRENT_BINARY_DIR}/gen/kernel_src.h)
add_custom_command(
    OUTPUT  ${KERNEL_GEN}
    COMMAND ${CMAKE_COMMAND} -DIN=${KERNEL_CL} -DOUT=${KERNEL_GEN}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_kernel.cmake
    DEPENDS ${KERNEL_CL} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_kernel.cmake
    VERBATIM)

if(APPLE)
  set(METAL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/kernels/tron_vanity_resident.metal)
  set(METAL_GEN ${CMAKE_CURRENT_BINARY_DIR}/gen/metal_kernel_src.h)
  set(METAL_HW_SRC ${CMAKE_CURRENT_SOURCE_DIR}/kernels/metal_hardware_profile.metal)
  set(METAL_HW_GEN ${CMAKE_CURRENT_BINARY_DIR}/gen/metal_hardware_profile_src.h)
  add_custom_command(
    OUTPUT ${METAL_GEN}
    COMMAND ${CMAKE_COMMAND} -DIN=${METAL_SRC} -DOUT=${METAL_GEN}
            -DVAR=kMetalKernelSource
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_kernel.cmake
    DEPENDS ${METAL_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_kernel.cmake
    VERBATIM)
  add_custom_command(
    OUTPUT ${METAL_HW_GEN}
    COMMAND ${CMAKE_COMMAND} -DIN=${METAL_HW_SRC} -DOUT=${METAL_HW_GEN}
            -DVAR=kMetalHardwareProfileSource
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_kernel.cmake
    DEPENDS ${METAL_HW_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_kernel.cmake
    VERBATIM)
  set(METAL_LIBS)
  foreach(METAL_RNG_MODE IN ITEMS 1 2 3)
    set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/tron_vanity_resident_${METAL_RNG_MODE}.air)
    set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/tron_vanity_resident_${METAL_RNG_MODE}.metallib)
    add_custom_command(
      OUTPUT ${METAL_LIB}
      COMMAND xcrun metal -DRESIDENT=1 -DRESIDENT_RNG=${METAL_RNG_MODE} -DECW=8 -DECBITS=256
              -DKPI=32 -DRESIDENT_SCALAR_KECCAK=1 -DMONT_N=1 -c ${METAL_SRC} -o ${METAL_AIR}
      COMMAND xcrun metallib ${METAL_AIR} -o ${METAL_LIB}
      DEPENDS ${METAL_SRC}
      VERBATIM)
    list(APPEND METAL_LIBS ${METAL_LIB})
  endforeach()
  # Also compile the largest production-tested point-walk batch so CI/builds
  # catch compiler regressions that the KPI=32 smoke variants cannot expose.
  set(METAL_TUNED_AIR ${CMAKE_CURRENT_BINARY_DIR}/tron_vanity_resident_kpi512.air)
  set(METAL_TUNED_LIB ${CMAKE_CURRENT_BINARY_DIR}/tron_vanity_resident_kpi512.metallib)
  add_custom_command(
    OUTPUT ${METAL_TUNED_LIB}
    COMMAND xcrun metal -DRESIDENT=1 -DRESIDENT_RNG=1 -DECW=8 -DECBITS=256
            -DKPI=512 -DRESIDENT_SCALAR_KECCAK=1 -DMONT_N=1 -c ${METAL_SRC}
            -o ${METAL_TUNED_AIR}
    COMMAND xcrun metallib ${METAL_TUNED_AIR} -o ${METAL_TUNED_LIB}
    DEPENDS ${METAL_SRC}
    VERBATIM)
  list(APPEND METAL_LIBS ${METAL_TUNED_LIB})
  set(METAL_HW_AIR ${CMAKE_CURRENT_BINARY_DIR}/metal_hardware_profile.air)
  set(METAL_HW_LIB ${CMAKE_CURRENT_BINARY_DIR}/metal_hardware_profile.metallib)
  add_custom_command(
    OUTPUT ${METAL_HW_LIB}
    COMMAND xcrun metal -c ${METAL_HW_SRC} -o ${METAL_HW_AIR}
    COMMAND xcrun metallib ${METAL_HW_AIR} -o ${METAL_HW_LIB}
    DEPENDS ${METAL_HW_SRC}
    VERBATIM)
  add_custom_target(metal_shader_check ALL DEPENDS ${METAL_GEN} ${METAL_HW_GEN}
                    ${METAL_LIBS} ${METAL_HW_LIB})
endif()

# ---- 主程序 ----
file(GLOB SRC CONFIGURE_DEPENDS src/*.cpp)
if(APPLE)
  list(APPEND SRC src/metal_backend.mm src/metal_hardware_profile.mm)
endif()
if(APPLE)
  add_executable(tron_vanity_generator ${SRC} ${KERNEL_GEN} ${METAL_GEN} ${METAL_HW_GEN})
else()
  add_executable(tron_vanity_generator ${SRC} ${KERNEL_GEN})
endif()
target_include_directories(tron_vanity_generator PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/gen)
target_link_libraries(tron_vanity_generator PRIVATE secp256k1 ${CMAKE_DL_LIBS})
if(TRON_ENABLE_CUDA)
  target_sources(tron_vanity_generator PRIVATE ${CUDA_GEN})
  target_compile_definitions(tron_vanity_generator PRIVATE TRON_ENABLE_CUDA=1)
endif()

find_package(Threads REQUIRED)
target_link_libraries(tron_vanity_generator PRIVATE Threads::Threads)

include(CTest)
if(BUILD_TESTING)
  add_executable(run_config_test tests/run_config_test.cpp src/run_config.cpp)
  target_include_directories(run_config_test PRIVATE src)
  add_test(NAME run_config COMMAND run_config_test)
  foreach(CUDA_TEST_RNG IN ITEMS 1 2 3)
    add_executable(cuda_kernel_test_${CUDA_TEST_RNG} tests/cuda_kernel_test.cpp
      src/crypto.cpp src/keccak.cpp)
    target_include_directories(cuda_kernel_test_${CUDA_TEST_RNG} PRIVATE src)
    target_compile_definitions(cuda_kernel_test_${CUDA_TEST_RNG} PRIVATE RESIDENT_RNG=${CUDA_TEST_RNG})
    target_link_libraries(cuda_kernel_test_${CUDA_TEST_RNG} PRIVATE secp256k1)
    add_test(NAME cuda_kernel_rng_${CUDA_TEST_RNG} COMMAND cuda_kernel_test_${CUDA_TEST_RNG})
    add_executable(opencl_pair_math_test_${CUDA_TEST_RNG} tests/cuda_kernel_test.cpp src/crypto.cpp src/keccak.cpp)
    target_include_directories(opencl_pair_math_test_${CUDA_TEST_RNG} PRIVATE src)
    target_compile_definitions(opencl_pair_math_test_${CUDA_TEST_RNG} PRIVATE RESIDENT_PAIR_INVERSE=1 RESIDENT_RNG=${CUDA_TEST_RNG})
    target_link_libraries(opencl_pair_math_test_${CUDA_TEST_RNG} PRIVATE secp256k1)
    add_test(NAME opencl_pair_math_${CUDA_TEST_RNG} COMMAND opencl_pair_math_test_${CUDA_TEST_RNG})
  endforeach()
  add_executable(opencl_runtime_test tests/opencl_runtime_test.cpp ${KERNEL_GEN}
    src/ocl.cpp src/resident_backend.cpp src/cuda_driver.cpp src/crypto.cpp
    src/keccak.cpp src/rng.cpp src/dictionary.cpp)
  target_include_directories(opencl_runtime_test PRIVATE src ${CMAKE_CURRENT_BINARY_DIR}/gen)
  target_link_libraries(opencl_runtime_test PRIVATE secp256k1 Threads::Threads ${CMAKE_DL_LIBS})
  if(WIN32)
    target_link_libraries(opencl_runtime_test PRIVATE bcrypt)
  endif()
  add_test(NAME opencl_runtime COMMAND opencl_runtime_test)
  set_tests_properties(opencl_runtime PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
endif()

if(APPLE)
  target_link_libraries(tron_vanity_generator PRIVATE
    "-framework Foundation" "-framework Metal")
  set_source_files_properties(src/metal_backend.mm PROPERTIES
    COMPILE_FLAGS "-fobjc-arc")
  target_compile_definitions(tron_vanity_generator PRIVATE TRON_METAL_MIN_MACOS=26)
endif()

if(WIN32)
  target_link_libraries(tron_vanity_generator PRIVATE bcrypt)
endif()

if(MSVC)
  target_compile_options(tron_vanity_generator PRIVATE /EHsc /utf-8
      $<$<CONFIG:Release>:/O2>)
else()
  # 不用 -march=native：便于把 exe 复制到 CPU 指令集不同的机器
  target_compile_options(tron_vanity_generator PRIVATE -O3)
endif()

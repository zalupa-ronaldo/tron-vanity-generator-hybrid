# 把 OpenCL 内核源码打包成一个 C 头文件里的 raw string。
# 用法: cmake -DIN=<in.cl> -DOUT=<out.h> -P embed_kernel.cmake
file(READ "${IN}" _src)
if(NOT VAR)
  set(VAR kGpuKernelSource)
endif()
file(WRITE "${OUT}"
"// 自动生成，勿手改。源: ${IN}\n#pragma once\nstatic const char* ${VAR} = R\"CLK(\n${_src}\n)CLK\";\n")

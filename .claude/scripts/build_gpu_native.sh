#!/bin/bash
# Build gpu_native binary against the freshly built libnntrainer.so on x86+OpenCL.
# Mirrors meson's include/define flag set (from compile_commands.json), adds the
# generated kernel-string header dir and the gpu_native source dir. No -DUSE__FP16.
set -x
B=/home/nntrainer/nntrainer/build_x86_cl
GN=/home/nntrainer/nntrainer/Applications/CausalLM/gpu_native
cd "$B" || exit 2

INCLUDES="-Inntrainer/libnntrainer.a.p -Inntrainer -I../nntrainer -Iapi -I../api -I../api/ccapi/include \
-Inntrainer/compiler -I../nntrainer/compiler -Inntrainer/schema -I../nntrainer/schema \
-Inntrainer/dataset -I../nntrainer/dataset -Inntrainer/layers/loss -I../nntrainer/layers/loss \
-Inntrainer/layers -I../nntrainer/layers -Inntrainer/models -I../nntrainer/models \
-Inntrainer/optimizers -I../nntrainer/optimizers \
-Inntrainer/tensor/cpu_backend/fallback -I../nntrainer/tensor/cpu_backend/fallback \
-Inntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl -I../nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl \
-Inntrainer/tensor/cpu_backend/ggml_interface -I../nntrainer/tensor/cpu_backend/ggml_interface \
-Inntrainer/tensor/cpu_backend/cblas_interface -I../nntrainer/tensor/cpu_backend/cblas_interface \
-Inntrainer/tensor/cpu_backend/x86 -I../nntrainer/tensor/cpu_backend/x86 \
-Inntrainer/tensor/cpu_backend -I../nntrainer/tensor/cpu_backend \
-Inntrainer/tensor/cl_operations -I../nntrainer/tensor/cl_operations \
-Inntrainer/tensor -I../nntrainer/tensor -Inntrainer/utils -I../nntrainer/utils \
-Inntrainer/graph -I../nntrainer/graph -Inntrainer/opencl -I../nntrainer/opencl \
-Inntrainer/layers/cl_layers -I../nntrainer/layers/cl_layers \
-I../subprojects/iniparser/src -Isubprojects/iniparser/__CMake_build -I../subprojects/iniparser/__CMake_build \
-Isubprojects/iniparser -I../subprojects/iniparser -I/usr/include/x86_64-linux-gnu/openblas-pthread/ \
-Inntrainer/tensor/cl_operations/cl_kernels -I$GN"

DEFINES="-D_FILE_OFFSET_BITS=64 -std=c++17 -O3 -march=native -mavx2 -mfma \
'-DVERSION=\"0.6.0\"' -DVERSION_MAJOR=0 -DVERSION_MINOR=6 -DVERSION_MICRO=0 \
-DMIN_CPP_VERSION=201703L -DMMAP_READ=1 -DKERNEL_CACHE=0 -DENABLE_FP16=1 -DUSE_MMAP=1 \
-DENABLE_OPENCL=1 -DCL_TARGET_OPENCL_VERSION=200 -DOPENCL_KERNEL_PATH=nntrainer_opencl_kernels \
-DML_API_COMMON=0 -DUSE_BLAS=1 -DHGEMM_EXPERIMENTAL_KERNEL=false -DNNTR_NUM_THREADS=4 \
-D__LOGGING__=1 -DREDUCE_TOLERANCE=1 '-DNNTRAINER_CONF_PATH=\"/usr/local/etc/nntrainer.ini\"' \
-fPIC -pthread -Wno-unused-variable -Wno-sign-compare"

eval c++ $INCLUDES $DEFINES \
  "$GN/main.cpp" "$GN/qwen3_forward.cpp" \
  -o /tmp/nntrainer_qwen3_gpu_x86 \
  -L"$B/nntrainer" -lnntrainer \
  /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so \
  -lOpenCL -ldl -lm -pthread \
  -Wl,-rpath,"$B/nntrainer"
echo "GXX_EXIT=$?"

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   per_layer_slice.cpp
 * @date   07 Apr 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Joonseok Oh <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Selects per-layer input chunk from packed per-layer embedding tensor.
 */

#include <cstdlib>
#include <cstring>
#include <env_compat.h>
#include <per_layer_slice.h>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_elementwise.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#endif

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

namespace {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
/**
 * @brief Is this pointer NOT dereferenceable by the host?
 *
 * @details True only for a plain cudaMalloc allocation. That is what the
 *          activation pool becomes under NNTR_CUDA_DEV_ACT (auto-armed on a
 *          discrete GPU with concurrentManagedAccess, cuda_context.cpp), and
 *          the host copy below would fault on it. Managed/UVM and host
 *          allocations both stay host-addressable and answer false.
 */
bool device_only(const void *p) {
  if (p == nullptr)
    return false;
  cudaPointerAttributes a{};
  const bool ok = cudaPointerGetAttributes(&a, p) == cudaSuccess;
  cudaGetLastError(); // a non-CUDA host pointer sets an error; clear it
  return ok && a.type == cudaMemoryTypeDevice;
}
#endif

/**
 * @brief Copy one contiguous slice row, wherever the two ends live.
 *
 * @details The fallback below is a pure copy, so it stays correct on a
 *          device-only activation pool as long as the copy engine does it
 *          instead of the CPU. std::memcpy on a cudaMalloc pointer is a
 *          SIGSEGV, which is what any run with the device slice-copy declined
 *          (NNTR_CUDA_ELTWISE=0) hit once this layer became constructible on
 *          the cuda context.
 */
void slice_row_copy(void *dst, const void *src, size_t bytes,
                    [[maybe_unused]] bool needs_device_copy) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  if (needs_device_copy) {
    // cudaMemcpyDefault infers the direction from the unified address space,
    // so this one call covers D2D / D2H / H2D without inspecting both ends.
    cudaMemcpy(dst, src, bytes, cudaMemcpyDefault);
    return;
  }
#endif
  std::memcpy(dst, src, bytes);
}
} // namespace

void PerLayerSliceLayer::finalize(nntrainer::InitLayerContext &context) {
  auto dims = context.getInputDimensions();
  auto in_dim = dims[0];
  if (!std::get<nntrainer::props::SkipPrefill>(slice_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(slice_props).get();

  unsigned int feature_size = std::get<props::FeatureSize>(slice_props).get();
  NNTR_THROW_IF(feature_size == 0, std::invalid_argument)
    << "feature_size must be > 0";
  NNTR_THROW_IF(in_dim.width() % feature_size != 0, std::invalid_argument)
    << "input width must be divisible by feature_size";

  auto out_dim = in_dim;
  out_dim.width(feature_size);
  context.setOutputDimensions({out_dim});
}

void PerLayerSliceLayer::forwarding(nntrainer::RunLayerContext &context,
                                    bool training) {}

void PerLayerSliceLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  // A chunked prefill calls this with from > 0 for every block after the first,
  // so "prefill" is any multi-token call, not just the from==0 one.
  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  auto &in = context.getInput(SINGLE_INOUT_IDX);
  auto &out = context.getOutput(SINGLE_INOUT_IDX);

  unsigned int feature_size = std::get<props::FeatureSize>(slice_props).get();
  unsigned int layer_index = std::get<props::LayerIndex>(slice_props).get();

  auto in_dim = in.getDim();
  unsigned int num_layers = in_dim.width() / feature_size;
  NNTR_THROW_IF(layer_index >= num_layers, std::invalid_argument)
    << "layer_index out of range";

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out.getDim();
  in_step_dim.batch(1);
  out_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.height(to - from);

  unsigned int b_size = in_dim.batch();
  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step = out.getSharedDataTensor(
      out_step_dim, b * out.getDim().getFeatureLen(), true);

    unsigned int tokens = in_step_dim.height();
    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      float *in_data = in_step.getData<float>();
      float *out_data = out_step.getData<float>();
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
      // Host memcpy slicing reads the GPU-produced UVM input on the CPU; sync
      // first in async mode (no-op in default sync mode).
      nntrainer::cuda::StreamManager::Global().finishIfAsync();
      // BOTH ends must be probed: the output is a separate pool tensor, so an
      // input-only probe (what the FP16 branch below used to do) misses the
      // half that actually faults on a write.
      const bool dev_copy = device_only(in_data) || device_only(out_data);
#else
      const bool dev_copy = false;
#endif
      for (unsigned int t = 0; t < tokens; ++t) {
        const float *src =
          in_data + t * in_dim.width() + layer_index * feature_size;
        float *dst = out_data + t * feature_size;
        slice_row_copy(dst, src, sizeof(float) * feature_size, dev_copy);
      }
#ifdef ENABLE_FP16
    } else if (in_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
      _FP16 *in_data = in_step.getData<_FP16>();
      _FP16 *out_data = out_step.getData<_FP16>();
      bool done = false;
      bool dev_copy = false;
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
      // GPU slice-copy: keep the packed per-layer embedding slice on-device
      // instead of the host memcpy loop. Opt-in (NNTR_CUDA_ELTWISE).
      static const bool gpu = nntr_env_on("NNTR_CUDA_ELTWISE");
      dev_copy = device_only(in_data) || device_only(out_data);
      if (gpu) {
        cudaPointerAttributes pa{};
        // Pinned host-mapped (zero-copy) memory reports Host but is
        // kernel-reachable via its devicePointer -- mirror cuda::dev_accessible
        // (cuda_context_manager.cpp:248-254). On an integrated GPU with
        // concurrentManagedAccess==0 (Tegra/Orin) every pool is allocated that
        // way, so a Managed||Device-only test leaves one host memcpy per
        // decoder layer inside the captured decode graph.
        bool dev =
          cudaPointerGetAttributes(&pa, in_data) == cudaSuccess &&
          (pa.type == cudaMemoryTypeManaged || pa.type == cudaMemoryTypeDevice ||
           (pa.type == cudaMemoryTypeHost && pa.devicePointer != nullptr));
        cudaGetLastError();
        if (dev && nntrainer::cuda::cuda_slice_copy_fp16(
                     reinterpret_cast<const unsigned short *>(in_data),
                     reinterpret_cast<unsigned short *>(out_data), tokens,
                     in_dim.width(), layer_index * feature_size, feature_size))
          done = true;
      }
      // The fallback reads whatever the device just produced.
      if (!done)
        nntrainer::cuda::StreamManager::Global().finishIfAsync();
#endif
      if (!done)
        for (unsigned int t = 0; t < tokens; ++t) {
          const _FP16 *src =
            in_data + t * in_dim.width() + layer_index * feature_size;
          _FP16 *dst = out_data + t * feature_size;
          slice_row_copy(dst, src, sizeof(_FP16) * feature_size, dev_copy);
        }
#endif
    } else {
      throw std::invalid_argument(
        "[PerLayerSlice] unsupported activation data type");
    }
  }
}

void PerLayerSliceLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  auto out_dim = input_dimensions[0];
  out_dim.width(std::get<props::FeatureSize>(slice_props).get());
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, out_dim);
}

void PerLayerSliceLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE
nntrainer::Layer *create_per_layer_slice_layer() {
  return new PerLayerSliceLayer();
}
void destroy_per_layer_slice_layer(nntrainer::Layer *layer) { delete layer; }
extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_per_layer_slice_layer, destroy_per_layer_slice_layer};
}
#endif

} // namespace causallm

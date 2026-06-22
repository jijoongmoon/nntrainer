// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cuda_fc_layer.cpp
 * @date   22 Jun 2026
 * @brief  Fully Connected Layer for the NVIDIA CUDA backend.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cuda_fc_layer.h>

#include <common_properties.h>
#include <layer_context.h>
#include <limits>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <cstdlib>

#include <cuda_blas_manager.h>
#include <cuda_fc_qint4.h>
#include <cuda_stream_manager.h>
#include <int4_tensor.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum FCParams { weight, bias };

namespace {
/**
 * @brief CUDA FC GEMM dispatch (correctness floor): FP32 -> cuBLAS on the UVM
 *        managed pointers; any other dtype -> host Tensor::dot (correct on the
 *        host-coherent managed memory). hidden_ = input_ * weight.
 */
// cuBLAS / fused kernels require device-accessible pointers. Weights/output are
// on the cuda context (UVM), but the input may come from a host (engine=cpu)
// layer across an engine boundary -- feeding a host-only pointer to a device
// kernel yields garbage. Only take a GPU path when the buffer is managed/device
// memory; otherwise fall through to the host path (correct on the UVM buffers).
static bool deviceAccessible(const void *p) {
  cudaPointerAttributes a{};
  bool ok = (cudaPointerGetAttributes(&a, p) == cudaSuccess) &&
            (a.type == cudaMemoryTypeManaged || a.type == cudaMemoryTypeDevice);
  cudaGetLastError(); // clear the benign error a host pointer may set
  return ok;
}

void cudaFcGemm(Tensor &input_, Tensor &weight, Tensor &hidden_) {
  using DT = ml::train::TensorDim::DataType;
  const DT wt = weight.getDataType();
  const DT at = input_.getDataType();

  const auto &id = input_.getDim();
  const auto &od = hidden_.getDim();
  // FC flattens all leading dims into M; width is the contracted dim K, the
  // output width is N (== unit). Data is contiguous row-major.
  const int K = (int)id.width();
  const int N = (int)od.width();
  const int M = (int)(id.batch() * id.channel() * id.height());

  static const bool fc_dbg = std::getenv("NNTR_FC_DEBUG") != nullptr;
  if (fc_dbg) {
    auto ptype = [](const void *p) {
      cudaPointerAttributes a{};
      bool ok = cudaPointerGetAttributes(&a, p) == cudaSuccess;
      cudaGetLastError();
      if (!ok) return 'u'; // unregistered / error
      switch (a.type) {
      case cudaMemoryTypeManaged: return 'm';
      case cudaMemoryTypeDevice: return 'd';
      case cudaMemoryTypeHost: return 'h';
      default: return '0';
      }
    };
    fprintf(stderr,
            "[FCDBG] wt=%d at=%d M=%d N=%d K=%d in=%c w=%c out=%c\n", (int)wt,
            (int)at, M, N, K, ptype(input_.getData<float>()),
            ptype(weight.getData<uint8_t>()), ptype(hidden_.getData<float>()));
  }

  // QINT4 weight: fused dequant-GEMM on device. Decodes the int4 weight inline
  // from the KAI Section-A super-row layout (no dense FP32 weight buffer) ->
  // fits real-size (e2b) memory. FP32 act.
  //
  // A loaded QINT4 weight is ALWAYS KAI_QSI4CXP_4x4x32 in memory (Section-A
  // packing + fp16 per-channel scales -- see Int4QTensor::upgradeQScheme /
  // Int4Utils::packPlainToSectionA), so we read that layout directly. Default
  // ON: the host Tensor::dot path is NYI for QINT4 on x86 (KAI is ARM-only),
  // so the device kernel is the only correct x86/CUDA path. Set
  // NNTR_FC_CUDA_QINT4=0 to force the host fallback (correct only on ARM).
  // The older plain-layout kernel (cuda_fc_qint4_gemm_fp32) is kept as a
  // foundation for genuinely plain weights but is not on this path.
  if (wt == DT::QINT4 && at == DT::FP32 && M > 0 && N > 0 && K > 0) {
    static const bool seca_enabled = []() {
      const char *e = std::getenv("NNTR_FC_CUDA_QINT4");
      return !(e != nullptr && e[0] == '0'); // default ON
    }();
    if (seca_enabled && (int)weight.getDim().height() == K) {
      const float *X = input_.getData<float>();
      const uint8_t *W = weight.getData<uint8_t>();
      const uint16_t *S = weight.getScale<uint16_t>();
      float *Y = hidden_.getData<float>();
      // Zero-copy when the tensors are device-accessible (UVM); otherwise mirror
      // the host-heap tensors into device memory (the QINT4 host GEMM is NYI on
      // x86, so this device path is the only correct one there).
      const bool all_dev =
        deviceAccessible(X) && deviceAccessible(W) && deviceAccessible(Y);
      const bool ok =
        all_dev ? cuda::cuda_fc_qint4_sectionA_gemm_fp32(X, W, S, Y, (unsigned)M,
                                                         (unsigned)N, (unsigned)K)
                : cuda::cuda_fc_qint4_sectionA_gemm_fp32_resident(
                    X, W, S, Y, (unsigned)M, (unsigned)N, (unsigned)K);
      if (ok)
        return;
    }
    // disabled / failed -> fall through to the host path (NYI for QINT4 x86).
  }

  // FP32 weight: cuBLAS SGEMM on the UVM pointers.
  if (wt == DT::FP32 && at == DT::FP32 && M > 0 && N > 0 && K > 0 &&
      deviceAccessible(input_.getData<float>()) &&
      deviceAccessible(weight.getData<float>()) &&
      deviceAccessible(hidden_.getData<float>()) &&
      cuda::BlasManager::Global().sgemmRowMajor(
        M, N, K, input_.getData<float>(), weight.getData<float>(),
        hidden_.getData<float>())) {
    // GEMM is enqueued on the backend stream; drain before host consumers
    // (bias add_i / the next layer) read the managed output.
    cuda::StreamManager::Global().finish();
    return;
  }

  // Host fallback: correct for FP16 / Q4_x / Q6_K / cross-engine host input
  // (and any GPU-path failure). Works because engine=cuda tensors are
  // host-coherent (UVM).
  input_.dot(weight, hidden_, false, false);
}
} // namespace

CudaFcLayer::CudaFcLayer() : LayerImpl(), fc_props(props::Unit()) {
  weight_idx.fill(2);
}

void CudaFcLayer::finalize(InitLayerContext &context) {
  if (!std::get<props::SkipPrefill>(*layer_impl_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(*layer_impl_props).get();
  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  auto unit = std::get<props::Unit>(fc_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Fully connected layer takes only one input";

  std::vector<TensorDim> output_dims(1);

  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == Tformat::NCHW);
  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);

  // CausalLM lm_head: skip_prefill vocab projection, plan height=1 (see the CL
  // FC for the rationale -- avoids a vocab-wide dead activation plane).
  if (skip_prefill && context.getName() == "output_of_causallm") {
    static const bool keep_full = std::getenv("NNTR_LMHEAD_OUT_FULL") != nullptr;
    if (!keep_full)
      output_dims[0].height(1);
  }

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  TensorDim bias_dim(
    1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0001 : 0b0100);

  TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[FCParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[FCParams::bias] =
      context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                            1.0f, bias_decay, "bias", true);
  }
}

void CudaFcLayer::exportTo(Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(fc_props, method, this);
}

void CudaFcLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, fc_props);
  LayerImpl::setProperty(remain_props);
}

void CudaFcLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  cudaFcGemm(input_, weight, hidden_);

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
    hidden_.add_i(bias);
  }
}

void CudaFcLayer::incremental_forwarding(RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {
  if (skip_prefill && from == 0)
    return;

  // by-reference so a quantized weight keeps its instance across forwards.
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  TensorDim input_dim = input_.getDim();
  TensorDim hidden_dim = hidden_.getDim();

  TensorDim input_step_dim = input_dim;
  TensorDim hidden_step_dim = hidden_dim;

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  input_step_dim.height(to - from);
  hidden_step_dim.height(to - from);

  // @todo: only correct for batch size 1
  Tensor input_step = input_.getSharedDataTensor(input_step_dim, 0, true);
  Tensor hidden_step = hidden_.getSharedDataTensor(hidden_step_dim, 0, true);

  cudaFcGemm(input_step, weight, hidden_step);

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
    hidden_step.add_i(bias);
  }
}

void CudaFcLayer::calcDerivative(RunLayerContext &context) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  ret_.dot_deriv_wrt_1(weight, derivative_, false, false);
}

void CudaFcLayer::calcGradient(RunLayerContext &context) {
  Tensor &djdw = context.getWeightGrad(weight_idx[FCParams::weight]);
  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &djdb = context.getWeightGrad(weight_idx[FCParams::bias]);
    if (context.isGradientFirstAccess(weight_idx[FCParams::bias])) {
      derivative_.sum({0, 1, 2}, djdb);
    } else {
      Tensor t = derivative_.sum({0, 1, 2});
      djdb.add_i(t);
    }
  }

  input_.dot_deriv_wrt_2(
    djdw, derivative_, false, false,
    !context.isGradientFirstAccess(weight_idx[FCParams::weight]));
}

} // namespace nntrainer

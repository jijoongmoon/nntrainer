// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file	fc_layer_cl.cpp
 * @date	7 May 2024
 * @brief	This is Fully Connected Layer Class for Neural Network with OpenCl
 * implementation
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Debadri Samaddar <s.debadri@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <blas_kernel_interface.h>
#include <chrono>
#include <common_properties.h>
#include <cstdio>
#include <cstdlib>
#include <fc_layer_cl.h>
#include <layer_context.h>
#include <mutex>
#include <lazy_tensor.h>
#include <limits>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum FCParams { weight, bias };

// =============================================================================
// Env-gated self-contained FC dispatch profiler. Counts wall-clock between
// FC layer entry and exit, split by M (prefill vs decode), with a sub-bin for
// "FC body excluding dotCl_v8c kernel time". Activated via NNTR_FC_PROFILE=1.
// =============================================================================
namespace {
struct FcProfBin {
  long long ns_total = 0;
  long long calls = 0;
};
struct FcProf {
  FcProfBin pre, dec;     // total entry-to-exit wall time
  FcProfBin pre_pre_v8c;  // pre-dotCl_v8c time
  FcProfBin pre_post_v8c; // post-dotCl_v8c time
  ~FcProf() {
    if (pre.calls + dec.calls == 0) return;
    std::fprintf(stderr,
                 "\n[FC-PROF]  prefill (M>1) / decode (M=1) FullyConnectedLayerCl\n");
    auto dump = [&](const char *tag, const FcProfBin &b) {
      if (b.calls == 0) return;
      double ms = b.ns_total / 1.0e6;
      std::fprintf(stderr, "  %-22s  %10.2f ms  %8lld calls  (%.3f ms/call)\n",
                   tag, ms, b.calls,
                   b.calls > 0 ? ms / b.calls : 0.0);
    };
    dump("total prefill", pre);
    dump("total decode", dec);
    dump("pre-v8c prefill", pre_pre_v8c);
    dump("post-v8c prefill", pre_post_v8c);
  }
};
inline FcProf &fc_prof() {
  static FcProf p;
  return p;
}
inline std::mutex &fc_prof_mtx() {
  static std::mutex m;
  return m;
}
inline bool fc_prof_enabled() {
  static const bool on = std::getenv("NNTR_FC_PROFILE") != nullptr;
  return on;
}
// Pre-zero of the FC output plane before the GEMM. Functionally redundant on
// the v8c path (the cl_mem writer / resident SVM writer / host-bounce all
// overwrite the complete M x N region), and it costs ~34ms of host memset per
// 1K prefill across the 182 FC calls (NNTR_FC_PROFILE pre-v8c 34 -> 0.4ms
// when skipped, hot prefill ~690 -> ~712 TPS). BUT skipping it
// deterministically flips generation (2026-06-12: prompt_1k "<eos>" ->
// incoherent babble, profiler-independent, stable across runs): with the
// one-handle-per-offset CLMEM pool and the FC async-map windows, racy
// host-plane reads that used to observe deterministic zeros observe stale
// bytes instead -- same family as the FC_FLUSH submit-split and the
// host-mapped-SVM ZEROS findings. The memset stays until the visibility wall
// (kernel-chain cl_mem residency) removes those windows.
// NNTR_FC_OUTZERO=0 skips it (KNOWN-UNSAFE, for A/B only).
inline bool fc_out_zero_enabled() {
  // 2026-06-12 re-baseline: default OFF. The zero's only effect was feeding
  // deterministic bytes to racy undrained readers (a race-pattern anchor,
  // not math -- the drained-capture probe proved the no-zero path computes
  // identical values); the re-baselined outputs absorb the pattern change.
  // NNTR_FC_OUTZERO=1 restores the memset.
  static const bool on = []() {
    const char *e = std::getenv("NNTR_FC_OUTZERO");
    return e && e[0] == '1';
  }();
  return on;
}
} // namespace

FullyConnectedLayerCl::FullyConnectedLayerCl() :
  LayerImplCl(), fc_props(props::Unit()) {
  // weight_idx.fill(std::numeric_limits<unsigned>::max());
  weight_idx.fill(2);
}

void FullyConnectedLayerCl::finalize(InitLayerContext &context) {
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

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == Tformat::NCHW);
  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);

  // CausalLM lm_head (name "output_of_causallm"): an untied QINT4 vocab
  // projection (unit=vocab). It is skip_prefill, so it only ever computes the
  // last position (decode M=1) -- the prefill rows are never written. Planning
  // its output at the full graph-build height (INIT_SEQ_LEN) makes a vocab-wide
  // activation tensor (e.g. 1024*262144*2 = 512MB for Gemma4) that the pool
  // allocates as device cl_mem up front yet never uses past row 0, adding ~1GB
  // of resident-but-dead GPU memory -> global bandwidth pressure that uniformly
  // slows every decode kernel (measured: untied decode ~5.5 vs tied ~12.7 TPS,
  // with identical transformer weights). Force height=1, mirroring the tied
  // tie_word_embedding lm_head which does the same. Math-identical (rows>0 were
  // never produced). NNTR_LMHEAD_OUT_FULL restores the old full-height planning.
  if (skip_prefill && context.getName() == "output_of_causallm") {
    static const bool keep_full = std::getenv("NNTR_LMHEAD_OUT_FULL") != nullptr;
    if (!keep_full)
      output_dims[0].height(1);
  }

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  // @todo : This NCHW format setting is just temporal, it needs to be set by
  // global configuration
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

void FullyConnectedLayerCl::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(fc_props, method, this);
}

void FullyConnectedLayerCl::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, fc_props);
  LayerImpl::setProperty(remain_props);
}

void FullyConnectedLayerCl::forwarding(RunLayerContext &context,
                                       bool training) {

  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  if (fc_out_zero_enabled())
    hidden_.setZero();
  // v8c (paper 8/4/4, channel-wise QINT4) GPU path — env-gated via
  // NNTR_FC_INT8_GPU. dotCl_v8c returns false when not applicable
  // (env unset / weight not QINT4 / shape misaligned), in which case we
  // fall through. dotCl only knows FP32/FP16 weight bytes, so for
  // quantized weight types we route through Tensor::dot (FloatTensor's
  // dotQnK / dotQInteger CPU dispatch) instead of feeding QINT4 bytes
  // into dotCl which would reinterpret them as FP32 and produce NaNs.
  if (!dotCl_v8c(input_, weight, hidden_)) {
    // The fallbacks require the pre-zero unconditionally.
    if (!fc_out_zero_enabled())
      hidden_.setZero();
    // Static GPU_CLMEM residency: the fallbacks below read the input and
    // write the output through host/SVM pointers only. Bridge a resident
    // input down first and raise a resident output afterwards, so a v8c
    // rejection can never leave a GPU_CLMEM tensor's planes inconsistent.
    clmem_lower_cl(input_, 0);
    auto wt = weight.getDataType();
    if (wt == ml::train::TensorDim::DataType::QINT4 ||
        wt == ml::train::TensorDim::DataType::Q4_0 ||
        wt == ml::train::TensorDim::DataType::Q4_K ||
        wt == ml::train::TensorDim::DataType::Q6_K) {
      input_.dot(weight, hidden_, false, false);
    } else {
      dotCl(input_, weight, hidden_);
    }
    clmem_raise_cl(hidden_, 0);
  }

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
    hidden_.add_i(bias);
  }
}

void FullyConnectedLayerCl::incremental_forwarding(RunLayerContext &context,
                                                   unsigned int from,
                                                   unsigned int to,
                                                   bool training) {
  if (skip_prefill && from == 0)
    return;
  auto _fc_t0 = fc_prof_enabled() ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
  // Use the by-reference getWeight so QINT4 callers see the same Int4QTensor
  // instance across forwards. The Tensor::operator= path on QINT4 deep-clones
  // the Int4QTensor (see tensor.cpp:376) — that would wipe the assembled
  // KAI rhs_packed cache on every token and tank decode throughput.
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

  // @todo: set reset stride as false. This implementation only works when
  // batch size is 1
  Tensor input_step = input_.getSharedDataTensor(input_step_dim, 0, true);
  Tensor hidden_step = hidden_.getSharedDataTensor(hidden_step_dim, 0, true);

  if (fc_out_zero_enabled())
    hidden_step.setZero();

  auto _v8c_t0 = fc_prof_enabled() ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
  // Same dispatch as forwarding(): v8c GPU path first, then CPU
  // ggml route for quantized weights, else dotCl.
  if (!dotCl_v8c(input_step, weight, hidden_step)) {
    // The fallbacks require the pre-zero unconditionally (see forwarding()).
    if (!fc_out_zero_enabled())
      hidden_step.setZero();
    // Static GPU_CLMEM residency bridge (see forwarding()).
    clmem_lower_cl(input_step, 0);
    auto wt = weight.getDataType();
    if (wt == ml::train::TensorDim::DataType::QINT4 ||
        wt == ml::train::TensorDim::DataType::Q4_0 ||
        wt == ml::train::TensorDim::DataType::Q4_K ||
        wt == ml::train::TensorDim::DataType::Q6_K) {
      input_step.dot(weight, hidden_step, false, false);
    } else {
      dotCl(input_step, weight, hidden_step);
    }
    clmem_raise_cl(hidden_step, 0);
  }
  auto _v8c_t1 = fc_prof_enabled() ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
    hidden_step.add_i(bias);
  }

  if (fc_prof_enabled()) {
    auto _fc_t1 = std::chrono::steady_clock::now();
    long long total_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(_fc_t1 - _fc_t0)
        .count();
    long long pre_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(_v8c_t0 - _fc_t0)
        .count();
    long long post_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(_fc_t1 - _v8c_t1)
        .count();
    bool is_dec = (to - from) == 1;
    std::lock_guard<std::mutex> lk(fc_prof_mtx());
    auto &p = fc_prof();
    if (is_dec) {
      p.dec.ns_total += total_ns;
      ++p.dec.calls;
    } else {
      p.pre.ns_total += total_ns;
      ++p.pre.calls;
      p.pre_pre_v8c.ns_total += pre_ns;
      ++p.pre_pre_v8c.calls;
      p.pre_post_v8c.ns_total += post_ns;
      ++p.pre_post_v8c.calls;
    }
  }
}

void FullyConnectedLayerCl::read(std::ifstream &file,
                                 RunLayerContext &run_context, bool opt_var,
                                 ml::train::ExecutionMode mode, bool trainable,
                                 TensorDim::DataType defineWeightDataType,
                                 bool fsu, size_t start_offset,
                                 bool read_from_offset, int file_fd) {
  Layer::read(file, run_context, opt_var, mode, trainable,
              defineWeightDataType, fsu, start_offset, read_from_offset,
              file_fd);
  // Eager v8c weight build at load (see header). Not under FSU: the weight
  // data may be streamed back out, invalidating the host pointer the v8c
  // cache is keyed on.
  if (!opt_var && !fsu)
    dotCl_v8c_prebuild_weight(
      run_context.getWeight(weight_idx[FCParams::weight]));
}

void FullyConnectedLayerCl::read(ReadSource src, RunLayerContext &run_context,
                                 bool opt_var, ml::train::ExecutionMode mode,
                                 bool trainable,
                                 TensorDim::DataType defineWeightDataType,
                                 bool fsu, size_t start_offset,
                                 bool read_from_offset, int file_fd) {
  Layer::read(src, run_context, opt_var, mode, trainable, defineWeightDataType,
              fsu, start_offset, read_from_offset, file_fd);
  if (!opt_var && !fsu)
    dotCl_v8c_prebuild_weight(
      run_context.getWeight(weight_idx[FCParams::weight]));
}

void FullyConnectedLayerCl::calcDerivative(RunLayerContext &context) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);

  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  ret_.dot_deriv_wrt_1(weight, derivative_, false, false);
}

void FullyConnectedLayerCl::calcGradient(RunLayerContext &context) {
  Tensor &djdw = context.getWeightGrad(weight_idx[FCParams::weight]);

  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &djdb = context.getWeightGrad(weight_idx[FCParams::bias]);

    if (context.isGradientFirstAccess(weight_idx[FCParams::bias])) {
      derivative_.sum({0, 1, 2}, djdb);
    } else {
      /// @todo optimize below by adding beta to Tensor::sum
      Tensor t = derivative_.sum({0, 1, 2});
      djdb.add_i(t);
    }
  }

  input_.dot_deriv_wrt_2(
    djdw, derivative_, false, false,
    !context.isGradientFirstAccess(weight_idx[FCParams::weight]));
}

} /* namespace nntrainer */

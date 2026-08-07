// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gated_delta_net_layer.cpp
 * @brief  Qwen3-Next Gated DeltaNet mixer — CPU forward (port of P1 reference).
 */

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <gated_delta_net_layer.h>
#include <node_exporter.h>
#include <stdexcept>
#include <vector>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cstdlib>
#include <cuda_context_manager.h>
#include <cuda_fc_dense.h>
#include <cuda_gdn.h>
#endif

// The retired lane's per-token GDN device kernel (nntrainer/cuda/cuda_gdn.*) is
// deliberately NOT carried over. Its prefill twin was a SEQUENTIAL scan, which
// was that lane's prefill bottleneck; the replacement is a chunked scan (see
// the FlashQLA-derived decomposition: cumsum g -> UT transform (I+A)^-1 ->
// per-chunk state carry -> o = inter + intra), whose decode case collapses to
// the same single-step recurrence the host path below already implements
// bit-exactly. Until those kernels exist this layer is host-only, which is
// correct-but-slow and, importantly, VISIBLE: it shows up in [CAP-AUDIT] rather
// than silently producing wrong numbers inside a graph capture.
// The call site below was ported to this base ahead of the kernel; cuda_gdn.cpp
// has now been brought over too (it is FP16-dense throughout and touches none
// of the int4/QS4CX payload machinery this base changed, so the "do not port
// the retired lane's CUDA code" caveat does not apply to it).
#define NNTR_GDN_HAVE_CUDA_KERNELS 1

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

static inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static inline float siluf(float x) { return x * sigmoidf(x); }
static inline float softplusf(float x) {
  return x > 20.0f ? x : std::log1p(std::exp(x));
}

// dtype-aware fp32 read of the first n elements. The GDN math runs in fp32
// regardless of the stored dtype (FP32 tiny validation / FP16 35B deployment).
static void readAsF32(const nntrainer::Tensor &t, size_t n,
                      std::vector<float> &out) {
  out.resize(n);
  if (t.getDataType() == ml::train::TensorDim::DataType::FP32) {
    std::memcpy(out.data(), t.getData<float>(), n * sizeof(float));
  } else if (t.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    const _FP16 *p = t.getData<_FP16>();
    for (size_t i = 0; i < n; ++i)
      out[i] = static_cast<float>(p[i]);
#else
    throw std::runtime_error("GDN: FP16 tensor without ENABLE_FP16");
#endif
  } else {
    throw std::runtime_error("GDN: unsupported tensor dtype");
  }
}

// zero-copy fp32 Tensor view over a heap buffer (for OpenBLAS sgemm dots)
static nntrainer::Tensor wrapF32(const float *p, unsigned int h,
                                 unsigned int w) {
  return nntrainer::Tensor(
    nntrainer::TensorDim(1, 1, h, w,
                         nntrainer::TensorDim::TensorType(
                           nntrainer::Tformat::NCHW,
                           nntrainer::TensorDim::DataType::FP32)),
    const_cast<float *>(p));
}

GatedDeltaNetLayer::GatedDeltaNetLayer() :
  LayerImpl(),
  num_v_heads(0), num_k_heads(0), head_k_dim(0), head_v_dim(0), key_dim(0),
  value_dim(0), conv_dim(0), conv_kernel(0), hidden_size(0), eps(1e-6f),
  gdn_props(props::LinearNumValueHeads(), props::LinearNumKeyHeads(),
            props::LinearKeyHeadDim(), props::LinearValueHeadDim(),
            props::LinearConvKernelDim()),
  w_in_proj_qkv(0), w_in_proj_z(0), w_in_proj_b(0), w_in_proj_a(0), w_conv1d(0),
  w_A_log(0), w_dt_bias(0), w_norm(0), w_out_proj(0) {}

void GatedDeltaNetLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, gdn_props);
  nntrainer::LayerImpl::setProperty(remain_props);
}

void GatedDeltaNetLayer::exportTo(
  nntrainer::Exporter &exporter,
  const ml::train::ExportMethods &method) const {
  nntrainer::LayerImpl::exportTo(exporter, method);
  exporter.saveResult(gdn_props, method, this);
}

void GatedDeltaNetLayer::calcDerivative(nntrainer::RunLayerContext &) {
  throw std::runtime_error("GatedDeltaNetLayer does not support training");
}
void GatedDeltaNetLayer::calcGradient(nntrainer::RunLayerContext &) {
  throw std::runtime_error("GatedDeltaNetLayer does not support training");
}

void GatedDeltaNetLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "GatedDeltaNet layer only supports single input";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<nntrainer::props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  const auto &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  hidden_size = in_dim.width();

  num_v_heads = std::get<props::LinearNumValueHeads>(gdn_props).get();
  num_k_heads = std::get<props::LinearNumKeyHeads>(gdn_props).get();
  head_k_dim = std::get<props::LinearKeyHeadDim>(gdn_props).get();
  head_v_dim = std::get<props::LinearValueHeadDim>(gdn_props).get();
  conv_kernel = std::get<props::LinearConvKernelDim>(gdn_props).get();
  key_dim = head_k_dim * num_k_heads;
  value_dim = head_v_dim * num_v_heads;
  conv_dim = key_dim * 2 + value_dim;

  context.setOutputDimensions({in_dim});

  const auto wt = context.getWeightDataType();
  auto WD = [&](unsigned h, unsigned w) {
    return nntrainer::TensorDim(
      1, 1, h, w, nntrainer::TensorDim::TensorType(context.getFormat(), wt),
      0b0011);
  };
  auto reqW = [&](const nntrainer::TensorDim &d, const std::string &n) {
    return context.requestWeight(d, weight_initializer, weight_regularizer,
                                 weight_regularizer_constant, weight_decay, n,
                                 true);
  };
  // projection weights stored [in, out] (HF [out,in] is transposed at load)
  w_in_proj_qkv = reqW(WD(hidden_size, conv_dim), "in_proj_qkv");
  w_in_proj_z = reqW(WD(hidden_size, value_dim), "in_proj_z");
  w_in_proj_b = reqW(WD(hidden_size, num_v_heads), "in_proj_b");
  w_in_proj_a = reqW(WD(hidden_size, num_v_heads), "in_proj_a");
  w_conv1d = reqW(WD(conv_dim, conv_kernel), "conv1d");   // [conv_dim, K]
  w_A_log = reqW(WD(1, num_v_heads), "A_log");
  w_dt_bias = reqW(WD(1, num_v_heads), "dt_bias");
  w_norm = reqW(WD(1, head_v_dim), "norm");
  w_out_proj = reqW(WD(value_dim, hidden_size), "out_proj");

  // persistent decode state (survives across incremental_forwarding calls)
  const unsigned int B = in_dim.batch();
  const auto fp32 = nntrainer::TensorDim::TensorType(
    context.getFormat(), nntrainer::TensorDim::DataType::FP32);
  state_idx = context.requestTensor(
    nntrainer::TensorDim(B, num_v_heads, head_k_dim, head_v_dim, fp32),
    "gdn_state", nntrainer::Initializer::ZEROS, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);
  conv_state_idx = context.requestTensor(
    nntrainer::TensorDim(B, 1, conv_dim, conv_kernel - 1, fp32),
    "gdn_conv_state", nntrainer::Initializer::ZEROS, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);

  // Prefill projection outputs, pooled. Height is the ALLOCATED input height,
  // not the prompt length -- runForward is handed the real length and writes a
  // prefix. All 30 GDN layers request byte-identical shapes, so the planner
  // hands them one set of slots rather than thirty.
  const unsigned int T_alloc = in_dim.height();
  auto reqProj = [&](unsigned int w, const std::string &n) {
    return context.requestTensor(nntrainer::TensorDim(B, 1, T_alloc, w, fp32),
                                 n, nntrainer::Initializer::NONE, false,
                                 nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  };
  proj_qkv_idx = reqProj(conv_dim, "gdn_proj_qkv");
  proj_z_idx = reqProj(value_dim, "gdn_proj_z");
  proj_b_idx = reqProj(num_v_heads, "gdn_proj_b");
  proj_a_idx = reqProj(num_v_heads, "gdn_proj_a");
}

void GatedDeltaNetLayer::ensureWeightCache(
  nntrainer::RunLayerContext &context) {
  if (wcache_loaded)
    return;
  readAsF32(context.getWeight(w_conv1d), (size_t)conv_dim * conv_kernel,
            wconv_f);
  readAsF32(context.getWeight(w_A_log), num_v_heads, alog_f);
  readAsF32(context.getWeight(w_dt_bias), num_v_heads, dtb_f);
  readAsF32(context.getWeight(w_norm), head_v_dim, wnorm_f);
  readAsF32(context.getWeight(w_in_proj_qkv), (size_t)hidden_size * conv_dim,
            wqkv_f);
  readAsF32(context.getWeight(w_in_proj_z), (size_t)hidden_size * value_dim,
            wz_f);
  readAsF32(context.getWeight(w_in_proj_b), (size_t)hidden_size * num_v_heads,
            wb_f);
  readAsF32(context.getWeight(w_in_proj_a), (size_t)hidden_size * num_v_heads,
            wa_f);
  readAsF32(context.getWeight(w_out_proj), (size_t)value_dim * hidden_size,
            wout_fv);
  wcache_loaded = true;
}

void GatedDeltaNetLayer::outProj(nntrainer::RunLayerContext &context,
                                 const float *normed, int B, int S,
                                 nntrainer::Tensor &output) {
  const int T = B * S, H = hidden_size, VAL = value_dim;
  nntrainer::Tensor normed_t(
    nntrainer::TensorDim(B, 1, S, VAL,
                         nntrainer::TensorDim::TensorType(
                           nntrainer::Tformat::NCHW,
                           nntrainer::TensorDim::DataType::FP32)),
    const_cast<float *>(normed));
  nntrainer::Tensor out_t =
    normed_t.dot(wrapF32(wout_fv.data(), VAL, H));
  if (output.getDataType() == ml::train::TensorDim::DataType::FP32) {
    std::memcpy(output.getData<float>(), out_t.getData<float>(),
                (size_t)T * H * sizeof(float));
  } else {
#ifdef ENABLE_FP16
    const float *src = out_t.getData<float>();
    _FP16 *dst = output.getData<_FP16>();
    for (size_t i = 0; i < (size_t)T * H; ++i)
      dst[i] = static_cast<_FP16>(src[i]);
#else
    throw std::runtime_error("GDN: FP16 output without ENABLE_FP16");
#endif
  }
  static const bool dbg = std::getenv("NNTR_GDN_DBG") != nullptr;
  if (dbg) {
    float nmax = 0, omax = 0;
    size_t nbad = 0;
    const float *op = out_t.getData<float>();
    for (size_t i = 0; i < (size_t)T * VAL; ++i)
      nmax = std::max(nmax, std::fabs(normed[i]));
    for (size_t i = 0; i < (size_t)T * H; ++i) {
      if (std::isnan(op[i]) || std::isinf(op[i]))
        ++nbad;
      else
        omax = std::max(omax, std::fabs(op[i]));
    }
    fprintf(stderr, "[gdn_dbg] %s outProj T=%d max|normed|=%g max|out|=%g bad=%zu\n",
            context.getName().c_str(), T, nmax, omax, nbad);
  }
}

void GatedDeltaNetLayer::forwarding(nntrainer::RunLayerContext &context,
                                    bool /*training*/) {
  runForward(context, context.getInput(SINGLE_INOUT_IDX).height(),
             /*save_state=*/false);
}

void GatedDeltaNetLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool /*training*/) {
  if (from == 0)
    // prefill: process the ACTUAL prompt length (to), not the padded tensor
    // height (INIT_SEQ_LEN); persist S + conv ring for decode.
    runForward(context, (int)to, /*save_state=*/true);
  else
    runDecode(context); // single-token decode using the persistent state
}

void GatedDeltaNetLayer::runForward(nntrainer::RunLayerContext &context,
                                    int seq_len, bool save_state) {
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  const int B = input.batch();
  const int S = seq_len; // actual sequence length (B==1: tensor stride == S)
  const int T = B * S;
  const int H = hidden_size;
  const int NVH = num_v_heads, NKH = num_k_heads;
  const int HKD = head_k_dim, HVD = head_v_dim;
  const int KEY = key_dim, VAL = value_dim, CONV = conv_dim, KS = conv_kernel;
  const int GQA = NVH / NKH;
  const float scale = 1.0f / std::sqrt((float)HKD);

  ensureWeightCache(context);
  const float *wconv = wconv_f.data();
  const float *A_log = alog_f.data();
  const float *dt_bias = dtb_f.data();
  const float *wnorm = wnorm_f.data();

  // projections: read only the T active input rows (B==1: rows 0..T-1; the
  // padded tail is skipped instead of multiplied), then fp32 sgemm over the
  // cached heap weights (see ensureWeightCache).
  // These four are the whole cost of GDN prefill. 25.3M params x 2 FLOP x T is
  // 1.01 TFLOP per layer at T=20000, against a recurrence that is ~3.5 TFLOP
  // for all 30 layers COMBINED -- so the scan being sequential matters far less
  // than these being on the host. They are fp16 dense, which cuBLAS takes
  // directly; the fp32-OUT variant keeps the accumulator cuBLAS already carries
  // so everything below, which is the fp32 reference, sees what it saw before.
  float *pq_w = context.getTensor(proj_qkv_idx).getData<float>();
  float *pz_w = context.getTensor(proj_z_idx).getData<float>();
  float *pb_w = context.getTensor(proj_b_idx).getData<float>();
  float *pa_w = context.getTensor(proj_a_idx).getData<float>();

  bool proj_gpu = false;
  bool pf_cmp = false;
  std::vector<float> pf_cmp_out, pf_cmp_state;
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
  static const int gdn_gpu_p = [] {
    const char *e = std::getenv("NNTR_CUDA_GDN");
    return e ? std::atoi(e) : 0;
  }();
  if (gdn_gpu_p > 0 &&
      input.getDataType() == ml::train::TensorDim::DataType::FP16) {
    const auto FP16D = ml::train::TensorDim::DataType::FP16;
    nntrainer::Tensor &Wqkv = context.getWeight(w_in_proj_qkv);
    nntrainer::Tensor &Wz = context.getWeight(w_in_proj_z);
    nntrainer::Tensor &Wb = context.getWeight(w_in_proj_b);
    nntrainer::Tensor &Wa = context.getWeight(w_in_proj_a);
    const void *xp = input.getData<_FP16>();
    if (Wqkv.getDataType() == FP16D && Wz.getDataType() == FP16D &&
        Wb.getDataType() == FP16D && Wa.getDataType() == FP16D &&
        nntrainer::cuda::dev_accessible(xp) &&
        nntrainer::cuda::dev_accessible(pq_w) &&
        nntrainer::cuda::dev_accessible(pz_w) &&
        nntrainer::cuda::dev_accessible(pb_w) &&
        nntrainer::cuda::dev_accessible(pa_w)) {
      using nntrainer::cuda::cuda_fc_dense_gemm_fp16_f32out;
      const unsigned uT = (unsigned)T, uH = (unsigned)H;
      // & not && on purpose: a partial failure must not leave some outputs
      // written and others stale, and the host fallback rewrites all four.
      proj_gpu =
        (int)cuda_fc_dense_gemm_fp16_f32out(xp, Wqkv.getData<_FP16>(), pq_w, uT,
                                            (unsigned)CONV, uH) &
        (int)cuda_fc_dense_gemm_fp16_f32out(xp, Wz.getData<_FP16>(), pz_w, uT,
                                            (unsigned)VAL, uH) &
        (int)cuda_fc_dense_gemm_fp16_f32out(xp, Wb.getData<_FP16>(), pb_w, uT,
                                            (unsigned)NVH, uH) &
        (int)cuda_fc_dense_gemm_fp16_f32out(xp, Wa.getData<_FP16>(), pa_w, uT,
                                            (unsigned)NVH, uH);
    }
  }
#endif
  if (!proj_gpu) {
    std::vector<float> xin_v;
    readAsF32(input, (size_t)T * H, xin_v);
    nntrainer::Tensor xin = wrapF32(xin_v.data(), T, H);
    nntrainer::Tensor o_qkv = wrapF32(pq_w, T, CONV), o_z = wrapF32(pz_w, T, VAL);
    nntrainer::Tensor o_b = wrapF32(pb_w, T, NVH), o_a = wrapF32(pa_w, T, NVH);
    xin.dot(wrapF32(wqkv_f.data(), H, CONV), o_qkv);
    xin.dot(wrapF32(wz_f.data(), H, VAL), o_z);
    xin.dot(wrapF32(wb_f.data(), H, NVH), o_b);
    xin.dot(wrapF32(wa_f.data(), H, NVH), o_a);
  }
  const float *pq = pq_w; // [T,CONV] token-major
  const float *pz = pz_w; // [T,VAL]
  const float *pb = pb_w;
  const float *pa = pa_w;

  // The decode conv ring is written from the projection output either way, so
  // it is a lambda rather than a tail: the device path below returns early.
  auto save_ring = [&]() {
    if (!save_state)
      return;
    float *cs = context.getTensor(conv_state_idx).getData<float>();
    for (int bi = 0; bi < B; ++bi)
      for (int c = 0; c < CONV; ++c)
        for (int j = 0; j < KS - 1; ++j) {
          const int ti = S - (KS - 1) + j; // position feeding ring slot j
          cs[(bi * CONV + c) * (KS - 1) + j] =
            (ti < 0) ? 0.0f : pq[(bi * S + ti) * CONV + c];
        }
  };

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
  // Everything from the conv1d to out_proj, on the device. This is what makes
  // a long prefill possible at all: the host code below is ~3e11 serial FMA
  // for the scan alone at T=20000, plus 10 TFLOP of out_proj across the 30
  // layers at a measured 5.6 GFLOPS.
  if (proj_gpu && B == 1) {
    const auto FP16D = ml::train::TensorDim::DataType::FP16;
    nntrainer::Tensor &Wout = context.getWeight(w_out_proj);
    float *st = context.getTensor(state_idx).getData<float>();
    const float *rg = context.getTensor(conv_state_idx).getData<float>();
    if (Wout.getDataType() == FP16D && output.getDataType() == FP16D &&
        nntrainer::cuda::dev_accessible(Wout.getData<_FP16>()) &&
        nntrainer::cuda::dev_accessible(output.getData<_FP16>()) &&
        nntrainer::cuda::dev_accessible(st)) {
      if (nntrainer::cuda::cuda_gdn_prefill_fp16(
            pq_w, pz_w, pb_w, pa_w,
            reinterpret_cast<const unsigned short *>(Wout.getData<_FP16>()),
            wconv_f.data(), alog_f.data(), dtb_f.data(), wnorm_f.data(), st, rg,
            reinterpret_cast<unsigned short *>(output.getData<_FP16>()),
            (unsigned)T, (unsigned)H, (unsigned)NVH, (unsigned)NKH,
            (unsigned)HKD, (unsigned)HVD, (unsigned)KS, eps,
            /*seed_state=*/false, save_state)) {
        if (gdn_gpu_p < 2) {
          save_ring();
          return;
        }
        // NNTR_CUDA_GDN=2: keep the device result and fall through to the host
        // reference, then diff. The host scan always starts from S=0, so no
        // state restore is needed; it overwrites both output and state with the
        // reference values, which is what we want to keep.
        pf_cmp_out.resize((size_t)T * H);
        const _FP16 *o16 = output.getData<_FP16>();
        for (size_t i = 0; i < (size_t)T * H; ++i)
          pf_cmp_out[i] = static_cast<float>(o16[i]);
        pf_cmp_state.assign(st, st + (size_t)NVH * HKD * HVD);
        pf_cmp = true;
      }
      // A failure here mutated nothing the host path cannot recompute: the
      // scan writes `state` only on its success path and `output` is fully
      // rewritten below.
    }
  }
#endif

  // causal depthwise conv1d + SiLU (per sequence, left-pad K-1)
  std::vector<float> conv(T * CONV);
  for (int bi = 0; bi < B; ++bi)
    for (int c = 0; c < CONV; ++c)
      for (int t = 0; t < S; ++t) {
        float acc = 0.0f;
        for (int j = 0; j < KS; ++j) {
          int ti = t - (KS - 1) + j;
          acc += wconv[c * KS + j] *
                 (ti < 0 ? 0.0f : pq[(bi * S + ti) * CONV + c]);
        }
        conv[(bi * S + t) * CONV + c] = siluf(acc);
      }

  // split [q|k|v] + GQA repeat; beta/g
  std::vector<float> q(T * NVH * HKD), k(T * NVH * HKD), v(T * NVH * HVD);
  std::vector<float> beta(T * NVH), gg(T * NVH);
  for (int i = 0; i < T; ++i)
    for (int vh = 0; vh < NVH; ++vh) {
      const int kh = vh / GQA;
      for (int d = 0; d < HKD; ++d) {
        q[(i * NVH + vh) * HKD + d] = conv[i * CONV + kh * HKD + d];
        k[(i * NVH + vh) * HKD + d] = conv[i * CONV + KEY + kh * HKD + d];
      }
      for (int d = 0; d < HVD; ++d)
        v[(i * NVH + vh) * HVD + d] = conv[i * CONV + 2 * KEY + vh * HVD + d];
      beta[i * NVH + vh] = sigmoidf(pb[i * NVH + vh]);
      gg[i * NVH + vh] =
        -std::exp(A_log[vh]) * softplusf(pa[i * NVH + vh] + dt_bias[vh]);
    }

  // l2norm(q,k) over head dim
  auto l2 = [&](std::vector<float> &a) {
    for (int i = 0; i < T * NVH; ++i) {
      float *r = &a[i * HKD], ss = 0.0f;
      for (int d = 0; d < HKD; ++d)
        ss += r[d] * r[d];
      const float inv = 1.0f / std::sqrt(ss + eps);
      for (int d = 0; d < HKD; ++d)
        r[d] *= inv;
    }
  };
  l2(q);
  l2(k);

  // decay-first delta recurrence (per batch, per v-head)
  float *state = save_state
                   ? context.getTensor(state_idx).getData<float>()
                   : nullptr; // [B,NVH,HKD,HVD]
  std::vector<float> core(T * NVH * HVD, 0.0f);
  for (int bi = 0; bi < B; ++bi)
    for (int vh = 0; vh < NVH; ++vh) {
      std::vector<float> Sh(HKD * HVD, 0.0f);
      for (int t = 0; t < S; ++t) {
        const int tok = bi * S + t;
        const float gt = std::exp(gg[tok * NVH + vh]);
        const float bt = beta[tok * NVH + vh];
        const float *qr = &q[(tok * NVH + vh) * HKD];
        const float *kr = &k[(tok * NVH + vh) * HKD];
        const float *vr = &v[(tok * NVH + vh) * HVD];
        for (int idx = 0; idx < HKD * HVD; ++idx)
          Sh[idx] *= gt;
        std::vector<float> kv(HVD, 0.0f);
        for (int a = 0; a < HKD; ++a)
          for (int b = 0; b < HVD; ++b)
            kv[b] += Sh[a * HVD + b] * kr[a];
        std::vector<float> dl(HVD);
        for (int b = 0; b < HVD; ++b)
          dl[b] = (vr[b] - kv[b]) * bt;
        for (int a = 0; a < HKD; ++a)
          for (int b = 0; b < HVD; ++b)
            Sh[a * HVD + b] += kr[a] * dl[b];
        float *o = &core[(tok * NVH + vh) * HVD];
        for (int b = 0; b < HVD; ++b)
          o[b] = 0.0f;
        for (int a = 0; a < HKD; ++a) {
          const float qq = qr[a] * scale;
          for (int b = 0; b < HVD; ++b)
            o[b] += Sh[a * HVD + b] * qq;
        }
      }
      if (save_state) // persist final recurrent state for decode
        std::memcpy(&state[(bi * NVH + vh) * HKD * HVD], Sh.data(),
                    (size_t)HKD * HVD * sizeof(float));
    }

  // gated RMSNorm: rmsnorm(core) * norm_weight * silu(z)  (over head_v_dim)
  std::vector<float> normed(T * VAL);
  for (int i = 0; i < T; ++i)
    for (int vh = 0; vh < NVH; ++vh) {
      const float *cr = &core[(i * NVH + vh) * HVD];
      const float *zr = &pz[i * VAL + vh * HVD];
      float var = 0.0f;
      for (int d = 0; d < HVD; ++d)
        var += cr[d] * cr[d];
      var /= HVD;
      const float inv = 1.0f / std::sqrt(var + eps);
      for (int d = 0; d < HVD; ++d)
        normed[i * VAL + vh * HVD + d] = cr[d] * inv * wnorm[d] * siluf(zr[d]);
    }

  // out_proj: [T,VAL] @ Wout[VAL,H] -> [T,H]. ALWAYS in fp32 (sgemm): the
  // fp16 hgemm accumulates in fp16 and can overflow to ±inf/NaN on large
  // normed rows. FP16 models use a cached fp32 clone of Wout.
  outProj(context, normed.data(), B, S, output);

  // persist the last (K-1) conv inputs (mixed_qkv) as the decode conv ring
  save_ring();

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
  if (pf_cmp) {
    // Report BOTH the layer output and the recurrent state. The output alone
    // is not enough: a scan that drifts only in S produces a fine-looking
    // prefill and then wrong decode, which is the failure this whole path is
    // most likely to have.
    float md_o = 0.0f, md_s = 0.0f;
    const _FP16 *o16 = output.getData<_FP16>();
    for (size_t i = 0; i < (size_t)T * H; ++i)
      md_o = std::max(md_o, std::fabs(pf_cmp_out[i] - static_cast<float>(o16[i])));
    if (save_state && !pf_cmp_state.empty()) {
      const float *st = context.getTensor(state_idx).getData<float>();
      for (size_t i = 0; i < pf_cmp_state.size(); ++i)
        md_s = std::max(md_s, std::fabs(pf_cmp_state[i] - st[i]));
    }
    fprintf(stderr, "[gdn_pf_cmp] %s T=%d max|gpu-host| out=%g state=%g\n",
            context.getName().c_str(), T, md_o, md_s);
  }
#endif
}

// single-token decode: input [B,1,1,hidden]; uses + updates the persistent
// recurrent state S and conv ring buffer in place.
void GatedDeltaNetLayer::runDecode(nntrainer::RunLayerContext &context) {
  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  const int B = input.batch();
  const int H = hidden_size;
  const int NVH = num_v_heads, NKH = num_k_heads;
  const int HKD = head_k_dim, HVD = head_v_dim;
  const int KEY = key_dim, VAL = value_dim, CONV = conv_dim, KS = conv_kernel;
  const int GQA = NVH / NKH;
  const float scale = 1.0f / std::sqrt((float)HKD);

  ensureWeightCache(context);
  const float *wconv = wconv_f.data();
  const float *A_log = alog_f.data();
  const float *dt_bias = dtb_f.data();
  const float *wnorm = wnorm_f.data();
  float *state = context.getTensor(state_idx).getData<float>();      // [B,NVH,HKD,HVD]
  float *cs = context.getTensor(conv_state_idx).getData<float>();    // [B,CONV,K-1]

  bool cmp_gpu = false; // NNTR_CUDA_GDN=2: GPU vs host output compare (debug)
  std::vector<float> cmp_out;
#if NNTR_GDN_HAVE_CUDA_KERNELS && defined(ENABLE_CUDA) && ENABLE_CUDA == 1 &&  \
  defined(ENABLE_FP16)
  // NNTR_CUDA_GDN: 1 = run the decode step on the GPU (one stream drain
  // instead of ~7 host<->GPU transitions), 2 = run BOTH and print the diff.
  static const int gdn_gpu = [] {
    const char *e = std::getenv("NNTR_CUDA_GDN");
    return e ? std::atoi(e) : 0;
  }();
  if (gdn_gpu > 0 && B == 1 &&
      input.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      output.getDataType() == ml::train::TensorDim::DataType::FP16) {
    const auto FP16D = ml::train::TensorDim::DataType::FP16;
    nntrainer::Tensor &Wqkv = context.getWeight(w_in_proj_qkv);
    nntrainer::Tensor &Wz = context.getWeight(w_in_proj_z);
    nntrainer::Tensor &Wb = context.getWeight(w_in_proj_b);
    nntrainer::Tensor &Wa = context.getWeight(w_in_proj_a);
    nntrainer::Tensor &Wout = context.getWeight(w_out_proj);
    const unsigned short *xp =
      reinterpret_cast<const unsigned short *>(input.getData<_FP16>());
    unsigned short *op =
      reinterpret_cast<unsigned short *>(output.getData<_FP16>());
    if (Wqkv.getDataType() == FP16D && Wz.getDataType() == FP16D &&
        Wb.getDataType() == FP16D && Wa.getDataType() == FP16D &&
        Wout.getDataType() == FP16D && nntrainer::cuda::dev_accessible(xp) &&
        nntrainer::cuda::dev_accessible(op) &&
        nntrainer::cuda::dev_accessible(state) &&
        nntrainer::cuda::dev_accessible(cs)) {
      std::vector<float> s_bak, r_bak;
      if (gdn_gpu >= 2) { // snapshot persistent state for the host rerun
        s_bak.assign(state, state + (size_t)NVH * HKD * HVD);
        r_bak.assign(cs, cs + (size_t)CONV * (KS - 1));
      }
      const bool ok = nntrainer::cuda::cuda_gdn_decode_fp16(
        xp, reinterpret_cast<const unsigned short *>(Wqkv.getData<_FP16>()),
        reinterpret_cast<const unsigned short *>(Wz.getData<_FP16>()),
        reinterpret_cast<const unsigned short *>(Wb.getData<_FP16>()),
        reinterpret_cast<const unsigned short *>(Wa.getData<_FP16>()),
        reinterpret_cast<const unsigned short *>(Wout.getData<_FP16>()),
        wconv_f.data(), alog_f.data(), dtb_f.data(), wnorm_f.data(), state,
        cs, op, H, NVH, NKH, HKD, HVD, KS, eps);
      if (ok && gdn_gpu == 1)
        return;
      if (gdn_gpu >= 2) { // restore; the host path below reruns the step
        if (ok) {
          cmp_gpu = true;
          cmp_out.resize(H);
          const _FP16 *o16 = reinterpret_cast<const _FP16 *>(op);
          for (unsigned int i = 0; i < (unsigned int)H; ++i)
            cmp_out[i] = static_cast<float>(o16[i]);
        }
        std::memcpy(state, s_bak.data(), s_bak.size() * sizeof(float));
        std::memcpy(cs, r_bak.data(), r_bak.size() * sizeof(float));
      }
      // !ok with gdn_gpu==1 falls through to the host path; registration/
      // alloc failures happen before any state-mutating kernel launches.
    }
  }
#endif

  // 1 token/batch: widen the input row once, sgemm over cached heap weights
  std::vector<float> xin_v;
  readAsF32(input, (size_t)B * H, xin_v);
  nntrainer::Tensor xin = wrapF32(xin_v.data(), B, H);
  nntrainer::Tensor t_qkv = xin.dot(wrapF32(wqkv_f.data(), H, CONV));
  nntrainer::Tensor t_z = xin.dot(wrapF32(wz_f.data(), H, VAL));
  nntrainer::Tensor t_b = xin.dot(wrapF32(wb_f.data(), H, NVH));
  nntrainer::Tensor t_a = xin.dot(wrapF32(wa_f.data(), H, NVH));
  const float *pq = t_qkv.getData<float>();
  const float *pz = t_z.getData<float>();
  const float *pb = t_b.getData<float>();
  const float *pa = t_a.getData<float>();

  std::vector<float> normed(B * VAL);
  for (int bi = 0; bi < B; ++bi) {
    // causal conv1d with ring: out[c] = silu(sum_j w[c,j]*[ring[c,0..K-2], x[c]])
    std::vector<float> conv(CONV);
    for (int c = 0; c < CONV; ++c) {
      float acc = 0.0f;
      for (int j = 0; j < KS - 1; ++j)
        acc += wconv[c * KS + j] * cs[(bi * CONV + c) * (KS - 1) + j];
      acc += wconv[c * KS + (KS - 1)] * pq[bi * CONV + c];
      conv[c] = siluf(acc);
      // advance ring: drop oldest, append current input column
      for (int j = 0; j < KS - 2; ++j)
        cs[(bi * CONV + c) * (KS - 1) + j] =
          cs[(bi * CONV + c) * (KS - 1) + j + 1];
      cs[(bi * CONV + c) * (KS - 1) + (KS - 2)] = pq[bi * CONV + c];
    }
    // split + GQA + l2norm + beta/g, then one decay-first step per v-head
    std::vector<float> qh(HKD), kh_(HKD), vh_(HVD), kv(HVD), dl(HVD), o(HVD);
    for (int vh = 0; vh < NVH; ++vh) {
      const int kh = vh / GQA;
      for (int d = 0; d < HKD; ++d) {
        qh[d] = conv[kh * HKD + d];
        kh_[d] = conv[KEY + kh * HKD + d];
      }
      for (int d = 0; d < HVD; ++d)
        vh_[d] = conv[2 * KEY + vh * HVD + d];
      // l2norm q,k
      float sq = 0, sk = 0;
      for (int d = 0; d < HKD; ++d) { sq += qh[d] * qh[d]; sk += kh_[d] * kh_[d]; }
      const float iq = 1.0f / std::sqrt(sq + eps), ik = 1.0f / std::sqrt(sk + eps);
      for (int d = 0; d < HKD; ++d) { qh[d] *= iq; kh_[d] *= ik; }
      const float gt = std::exp(-std::exp(A_log[vh]) *
                                softplusf(pa[bi * NVH + vh] + dt_bias[vh]));
      const float bt = sigmoidf(pb[bi * NVH + vh]);
      float *Sh = &state[(bi * NVH + vh) * HKD * HVD];
      for (int idx = 0; idx < HKD * HVD; ++idx)
        Sh[idx] *= gt;
      std::fill(kv.begin(), kv.end(), 0.0f);
      for (int a = 0; a < HKD; ++a)
        for (int b = 0; b < HVD; ++b)
          kv[b] += Sh[a * HVD + b] * kh_[a];
      for (int b = 0; b < HVD; ++b)
        dl[b] = (vh_[b] - kv[b]) * bt;
      for (int a = 0; a < HKD; ++a)
        for (int b = 0; b < HVD; ++b)
          Sh[a * HVD + b] += kh_[a] * dl[b];
      std::fill(o.begin(), o.end(), 0.0f);
      for (int a = 0; a < HKD; ++a) {
        const float qq = qh[a] * scale;
        for (int b = 0; b < HVD; ++b)
          o[b] += Sh[a * HVD + b] * qq;
      }
      // gated RMSNorm
      const float *zr = &pz[bi * VAL + vh * HVD];
      float var = 0.0f;
      for (int d = 0; d < HVD; ++d)
        var += o[d] * o[d];
      var /= HVD;
      const float inv = 1.0f / std::sqrt(var + eps);
      for (int d = 0; d < HVD; ++d)
        normed[bi * VAL + vh * HVD + d] = o[d] * inv * wnorm[d] * siluf(zr[d]);
    }
  }
  // out_proj (fp32 sgemm, as in runForward)
  outProj(context, normed.data(), B, 1, output);

  if (cmp_gpu) {
    float md = 0.0f;
#ifdef ENABLE_FP16
    const _FP16 *o16 = output.getData<_FP16>();
    for (unsigned int i = 0; i < (unsigned int)H; ++i)
      md = std::max(md, std::fabs(cmp_out[i] - static_cast<float>(o16[i])));
#endif
    fprintf(stderr, "[gdn_gpu_cmp] %s max|gpu-host|=%g\n",
            context.getName().c_str(), md);
  }
}

} // namespace causallm

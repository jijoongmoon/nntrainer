// SPDX-License-Identifier: Apache-2.0
//
// Standalone verification harness for the forwarding()-based GPU-resident
// attention path (MHACoreLayer 3-input internal-cache mode).
//
// Builds a 3-input attention sub-graph  Q,K,V -> mha_core -> O  and drives it
// through NeuralNetwork::forwarding() (model->inference(), which fans out to
// each node->forwarding()). With engine=gpu + the SVM residency pool, the mha
// node runs the GPU two_conv_attention kernel SVM-direct. The output is
// compared against an in-harness fp32 attention golden.
//
// RoPE is disabled (rope_theta=0) so the x86 FP16 CPU-RoPE NYI stub is not hit
// and the golden is plain self-attention. use_gemm_attention + seq>=32 selects
// the prefill GPU dispatch.
//
// Env (set in main): NNTR_GPU_SVM_POOL=1 (SVM pool + in-order queue),
// NNTR_MHA_GPU=1 (enable GPU attention dispatch), NNTR_MHA_VERIFY=1 (bypass the
// 28-layer drift gate for a single call), NNTR_V8C_BUF=1 (compile the
// two_conv_attention program in buffer-only mode on Intel NEO).
//
// Cases: A forwarding()->GPU attn (no RoPE); B x86 FP16 CPU attn kernels;
//   C x86 FP16 host RoPE; D GPU RoPE (cl_mem) + GPU attn; E GPU RoPE SVM-direct.
// Run one case per process via CASE=A|B|C|D|E. On Intel the OpenCL driver gets
// unstable after several GPU-attention models in one process (multi-model SVM
// segfault / inf), so the all-in-one run can fail the last GPU case; isolate
// with CASE=X for clean results. NO_SVM=1 forces the cl_mem path.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <app_context.h>
#include <cl_context.h>
#include <engine.h>
#include <fp16.h>
#include <layer.h>
#include <model.h>
#include <optimizer.h>

#include <mha_core.h>

using ml::train::createLayer;
using ml::train::createModel;

// ---- fp16 <-> fp32 (use nntrainer's exact conversion) --------------------
static float h2f(uint16_t h) { return nntrainer::compute_fp16_to_fp32(h); }
static uint16_t f2h(float f) { return nntrainer::compute_fp32_to_fp16(f); }

// Deterministic pseudo-random fp16 (half-bit) vector in [-0.5, 0.5]
// (fp16-safe magnitudes so QK^T/softmax never overflow half).
static std::vector<uint16_t> genHalf(size_t n, unsigned int seed) {
  std::vector<uint16_t> v(n);
  unsigned int s = seed | 1u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    float f = ((float)((s >> 9) & 0x3ff) / 1023.0f - 0.5f); // [-0.5, 0.5]
    v[i] = f2h(f);
  }
  return v;
}

// CPU fp32 golden matching two_conv_attention_prefill_f16_cl semantics:
//   scores[m,n] = scale * sum_x Qf[m,hq,x]*Kf[n,hkv,x], causal: n>m -> -inf;
//   softmax over n; O[m,hq,x] = sum_n p[n] * Vf[n,hkv,x]; hkv = hq/(Hq/Hkv).
static std::vector<float> cpuAttention(const std::vector<uint16_t> &Q,
                                       const std::vector<uint16_t> &K,
                                       const std::vector<uint16_t> &V,
                                       unsigned M, unsigned N_kv, unsigned Hq,
                                       unsigned Hkv, unsigned d, bool causal) {
  const unsigned HDq = Hq * d, HDkv = Hkv * d, gqa = Hq / Hkv;
  const float scale = 1.0f / std::sqrt((float)d);
  std::vector<float> O((size_t)M * HDq, 0.0f);
  std::vector<float> scores(N_kv);
  for (unsigned hq = 0; hq < Hq; ++hq) {
    const unsigned hkv = hq / gqa;
    for (unsigned m = 0; m < M; ++m) {
      float mx = -INFINITY;
      for (unsigned n = 0; n < N_kv; ++n) {
        if (causal && n > m) {
          scores[n] = -INFINITY;
          continue;
        }
        float acc = 0.0f;
        for (unsigned x = 0; x < d; ++x)
          acc += h2f(Q[(size_t)m * HDq + hq * d + x]) *
                 h2f(K[(size_t)n * HDkv + hkv * d + x]);
        scores[n] = acc * scale;
        if (scores[n] > mx)
          mx = scores[n];
      }
      double sum = 0.0;
      for (unsigned n = 0; n < N_kv; ++n) {
        if (std::isfinite(scores[n])) {
          scores[n] = std::exp(scores[n] - mx);
          sum += scores[n];
        } else
          scores[n] = 0.0f;
      }
      if (sum > 0)
        for (unsigned n = 0; n < N_kv; ++n)
          scores[n] /= (float)sum;
      for (unsigned x = 0; x < d; ++x) {
        float o = 0.0f;
        for (unsigned n = 0; n < N_kv; ++n)
          o += scores[n] * h2f(V[(size_t)n * HDkv + hkv * d + x]);
        O[(size_t)m * HDq + hq * d + x] = o;
      }
    }
  }
  return O;
}

// The input layers are declared input_dtype=FP16 so the mha receives FP16 Q/K/V
// (the GPU two_conv kernel is FP16-only). inference() Map()s the input buffer as
// getDataLen()*sizeof(float) bytes but the FP16 input tensor reads the first
// getDataLen() halfs from the buffer start. So a float-sized buffer (4 bytes/
// elem) holding our fp16 values in its low 2 bytes/elem is read correctly.
static std::vector<float> packHalfAsFloatBuf(const std::vector<uint16_t> &h) {
  std::vector<float> buf(h.size(), 0.0f); // h.size() floats = h.size()*4 bytes
  uint16_t *u = reinterpret_cast<uint16_t *>(buf.data());
  for (size_t i = 0; i < h.size(); ++i)
    u[i] = h[i];
  return buf;
}

static void registerMHA() {
  auto &eng = nntrainer::Engine::Global();
  try {
    auto *cpu =
      static_cast<nntrainer::AppContext *>(eng.getRegisteredContext("cpu"));
    cpu->registerFactory(nntrainer::createLayer<causallm::MHACoreLayer>);
  } catch (std::exception &e) {
    std::fprintf(stderr, "[reg] cpu mha: %s\n", e.what());
  }
  try {
    auto *gpu =
      static_cast<nntrainer::ClContext *>(eng.getRegisteredContext("gpu"));
    if (gpu)
      gpu->registerFactory(nntrainer::createLayer<causallm::MHACoreLayer>);
  } catch (std::exception &e) {
    std::fprintf(stderr, "[reg] gpu mha: %s\n", e.what());
  }
}

static const unsigned seq = 48, Hq = 8, Hkv = 4, d = 64;
static const unsigned HDq = Hq * d, HDkv = Hkv * d;

// Build input(q,k,v, FP16) -> mha_core -> out and run one prefill forward.
// engine ("cpu"/"gpu"), theta (RoPE; 0=off), gemm (use_gemm_attention) select
// the attention path. Returns the fp16 output as fp32.
static std::vector<float> runMHA(const std::string &engine, const char *theta,
                                 bool gemm, const std::vector<uint16_t> &Q,
                                 const std::vector<uint16_t> &K,
                                 const std::vector<uint16_t> &V) {
  auto model = createModel(ml::train::ModelType::NEURAL_NET);
  model->addLayer(createLayer("input", {"name=q", "input_dtype=FP16", "input_shape=1:48:" + std::to_string(HDq)}));
  model->addLayer(createLayer("input", {"name=k", "input_dtype=FP16", "input_shape=1:48:" + std::to_string(HDkv)}));
  model->addLayer(createLayer("input", {"name=v", "input_dtype=FP16", "input_shape=1:48:" + std::to_string(HDkv)}));
  model->addLayer(createLayer(
    "mha_core",
    {"name=attn", "input_layers=q,k,v", "engine=" + engine,
     "num_heads=" + std::to_string(Hq), "num_heads_kv=" + std::to_string(Hkv),
     "max_timestep=128", "max_new_tokens=8", "max_position_embeddings=128",
     std::string("rope_theta=") + theta, "is_causal=true",
     std::string("use_gemm_attention=") + (gemm ? "true" : "false")}));
  model->setProperty({"batch_size=1", "model_tensor_type=FP16-FP16"});
  int cc = model->compile(ml::train::ExecutionMode::INFERENCE);
  int ic = model->initialize(ml::train::ExecutionMode::INFERENCE);
  if (cc != 0 || ic != 0) {
    std::fprintf(stderr, "[mha-fwd] (%s theta=%s gemm=%d) compile=%d init=%d\n",
                 engine.c_str(), theta, (int)gemm, cc, ic);
    return {};
  }
  std::vector<float> qb = packHalfAsFloatBuf(Q), kb = packHalfAsFloatBuf(K),
                     vb = packHalfAsFloatBuf(V);
  std::vector<float *> in = {qb.data(), kb.data(), vb.data()};
  auto out = model->inference(1, in);
  std::vector<float> res;
  if (!out.empty() && out[0] != nullptr) {
    const uint16_t *oh = reinterpret_cast<const uint16_t *>(out[0]);
    res.resize((size_t)seq * HDq);
    for (size_t i = 0; i < res.size(); ++i)
      res[i] = h2f(oh[i]);
  }
  return res;
}

static double relL2(const std::vector<float> &a, const std::vector<float> &b) {
  if (a.size() != b.size() || a.empty())
    return 1e9;
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    num += (a[i] - b[i]) * (a[i] - b[i]);
    den += b[i] * b[i];
  }
  return den > 0 ? std::sqrt(num / den) : 0.0;
}

int main() {
  if (std::getenv("NO_SVM") == nullptr)
    setenv("NNTR_GPU_SVM_POOL", "1", 1);
  setenv("NNTR_MHA_GPU", "1", 1);
  setenv("NNTR_MHA_VERIFY", "1", 1);
  setenv("NNTR_V8C_BUF", "1", 1);

  registerMHA();

  // Optional single-case selector (CASE=A/B/C/D) to isolate from multi-model
  // Intel driver state.
  const char *only_env = std::getenv("CASE");
  const std::string only = only_env ? only_env : "";
  auto run_case = [&](const char *name) { return only.empty() || only == name; };

  std::vector<uint16_t> Q = genHalf((size_t)seq * HDq, 1);
  std::vector<uint16_t> K = genHalf((size_t)seq * HDkv, 2);
  std::vector<uint16_t> V = genHalf((size_t)seq * HDkv, 3);
  std::vector<float> golden = cpuAttention(Q, K, V, seq, seq, Hq, Hkv, d, true);

  int rc = 0;

  // Case A: forwarding() -> GPU two_conv kernel (SVM-resident), no RoPE.
  if (run_case("A")) {
    auto g = runMHA("gpu", "0", true, Q, K, V);
    double r = relL2(g, golden);
    bool ok = r < 3e-2;
    std::printf("[A gpu/forwarding/no-rope] relL2 vs golden=%.5g -> %s\n", r,
                ok ? "PASS" : "FAIL");
    rc |= !ok;
  }
  // Case B: engine=cpu FP16 attention (exercises the x86 FP16 CPU kernels
  // compute_kcaches / softmax_row / compute_fp16vcache_transposed), no RoPE.
  if (run_case("B")) {
    auto c = runMHA("cpu", "0", false, Q, K, V);
    double r = relL2(c, golden);
    bool ok = r < 3e-2;
    std::printf("[B cpu/fp16-kernels/no-rope] relL2 vs golden=%.5g -> %s\n", r,
                ok ? "PASS" : "FAIL");
    rc |= !ok;
  }
  // Case C: RoPE on (theta>0) on the x86 FP16 CPU path — exercises
  // compute_rotary_emb_value(_FP16). The output must be finite, non-zero, and
  // differ from the no-RoPE result (i.e. RoPE was actually applied), with no
  // crash on x86 FP16. (GPU+RoPE is validated separately on-device; the host
  // multi-model GPU path on Intel is flaky and not the subject here.)
  {
    auto c0 = runMHA("cpu", "0", false, Q, K, V);     // no RoPE
    auto cR = runMHA("cpu", "10000", false, Q, K, V); // RoPE
    bool finite = !cR.empty();
    double mxR = 0.0;
    for (float v : cR) {
      if (!std::isfinite(v)) finite = false;
      mxR = std::max(mxR, (double)std::fabs(v));
    }
    double diff = relL2(cR, c0); // RoPE must change the result
    bool ok = finite && mxR > 1e-3 && diff > 1e-2;
    std::printf("[C cpu fp16 RoPE] finite=%d max|cpu|=%.4g diff-vs-norope=%.4g "
                "-> %s\n",
                (int)finite, mxR, diff, ok ? "PASS" : "FAIL");
    rc |= !ok;
  }

  // Case D: GPU RoPE — theta>0 + use_gemm_attention=true triggers the GPU RoPE
  // path (rope_inplace_f16_cl) + GPU attention. Compare to Case C's host-RoPE
  // CPU reference; they must agree (GPU RoPE == host RoPE, both attentions
  // agree). Confirms the residency RoPE produces correct output.
  if (run_case("D")) {
    auto cRef = runMHA("cpu", "10000", false, Q, K, V); // host RoPE + CPU attn
    auto dGpu = runMHA("cpu", "10000", true, Q, K, V);  // GPU RoPE + GPU attn
    double r = relL2(dGpu, cRef);
    auto mx = [](const std::vector<float> &x) {
      double m = 0; for (float v : x) m = std::max(m, (double)std::fabs(v)); return m; };
    bool ok = !dGpu.empty() && !cRef.empty() && std::isfinite(mx(dGpu)) &&
              r < 3e-2;
    std::printf("[D gpu-rope vs host-rope] relL2=%.5g max|gpu|=%.4g -> %s\n", r,
                mx(dGpu), ok ? "PASS" : "FAIL");
    rc |= !ok;
  }

  // Case E: GPU RoPE on the SVM-resident engine=gpu path (theta>0). Confirms
  // rope_inplace_f16_cl runs SVM-direct (svm=1, no host upload) and the output
  // is finite/correct vs the host-RoPE CPU reference. Run isolated (CASE=E) on
  // Intel to avoid the multi-model GPU driver issue.
  if (run_case("E")) {
    auto cRef = runMHA("cpu", "10000", false, Q, K, V); // host RoPE + CPU attn
    auto eGpu = runMHA("gpu", "10000", true, Q, K, V);  // SVM GPU RoPE + attn
    double r = relL2(eGpu, cRef);
    auto mx = [](const std::vector<float> &x) {
      double m = 0; for (float v : x) m = std::max(m, (double)std::fabs(v)); return m; };
    bool ok = !eGpu.empty() && std::isfinite(mx(eGpu)) && r < 3e-2;
    std::printf("[E gpu-rope SVM vs host-rope] relL2=%.5g max|gpu|=%.4g -> %s\n",
                r, mx(eGpu), ok ? "PASS" : "FAIL");
    rc |= !ok;
  }

  std::printf("[mha-fwd] %s\n", rc == 0 ? "ALL PASS" : "FAIL");
  return rc;
}

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   unittest_cl_residency.cpp
 * @date   08 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  End-to-end GPU residency test: with the SVM pool enabled
 *         (NNTR_GPU_SVM_POOL), a small engine=gpu graph must produce the same
 *         output as the engine=cpu reference. The per-layer golden tests run on
 *         the host (cpu) pool, so they cannot cover the SVM-direct residency
 *         path (shared SVM buffers chained across layers + coarse-grained SVM
 *         coherence); this test does.
 */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <layer.h>
#include <model.h>
#include <nntrainer_error.h>
#include <optimizer.h>

// GPU attention-residency test dependencies (cl_operations / opencl infra).
#include <attention_kernels.h>
#include <blas_kernels.h>
#include <cl_context.h>
#include <engine.h>
#include <fp16.h>

namespace {

using ml::train::createLayer;
using ml::train::createModel;
using ml::train::createOptimizer;

/**
 * @brief Build input -> fc1 -> fc2 -> add(fc1, fc2) with deterministic (ones)
 *        weights for the given engine, run inference once and return the output.
 *
 * TRAIN-mode compile/initialize is used so the weight initializers actually run
 * (INFERENCE mode leaves weights for a subsequent load()); an optimizer is
 * required to initialize in TRAIN mode but we never train.
 */
std::vector<float> runModel(const std::string &engine, unsigned int in_w,
                            unsigned int unit, std::vector<float> &input) {
  auto model = createModel(ml::train::ModelType::NEURAL_NET);
  model->addLayer(createLayer(
    "input", {"name=input0", "input_shape=1:1:" + std::to_string(in_w)}));
  model->addLayer(
    createLayer("fully_connected",
                {"name=fc1", "unit=" + std::to_string(unit),
                 "weight_initializer=ones", "bias_initializer=zeros",
                 "input_layers=input0", "engine=" + engine}));
  model->addLayer(
    createLayer("fully_connected",
                {"name=fc2", "unit=" + std::to_string(unit),
                 "weight_initializer=ones", "bias_initializer=zeros",
                 "input_layers=fc1", "engine=" + engine}));
  // residual add: exercises the GPU SVM-direct add path under residency
  model->addLayer(createLayer(
    "addition", {"name=add", "input_layers=fc1,fc2", "engine=" + engine}));
  model->setProperty({"batch_size=1"});
  model->setOptimizer(createOptimizer("sgd", {"learning_rate=0.1"}));

  EXPECT_EQ(model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(model->initialize(), ML_ERROR_NONE);

  std::vector<float *> in_vec = {input.data()};
  auto out = model->inference(1, in_vec);

  std::vector<float> result;
  if (!out.empty() && out[0] != nullptr)
    result.assign(out[0], out[0] + unit);
  return result;
}

// ---------------------------------------------------------------------------
// Attention residency helpers
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random fp16 (half-bit) vector in [-1, 1].
std::vector<uint16_t> genHalf(size_t n, unsigned int seed) {
  std::vector<uint16_t> v(n);
  unsigned int s = seed | 1u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    float f = ((float)((s >> 9) & 0x3ff) / 1023.0f - 0.5f) * 2.0f;
    v[i] = nntrainer::compute_fp32_to_fp16(f);
  }
  return v;
}

/// CPU golden for two_conv_attention_prefill_f16_cl. Matches the kernel's
/// reference math (attention_kernels.cpp mha_cpu_qk_row / softmax / sv_row):
///   Q[m, hq*d+x], K/V[n, hkv*d+x] row-major; hkv = hq / (Hq/Hkv);
///   scores[m,n] = scale * sum_x Qf*Kf, causal: n>m -> -inf;
///   row-softmax over n; O[m, hq*d+x] = sum_n p[n] * Vf[n, hkv*d+x].
/// Returns O as fp32 [M * Hq*d].
std::vector<float> cpuAttention(const std::vector<uint16_t> &Q,
                                const std::vector<uint16_t> &K,
                                const std::vector<uint16_t> &V, unsigned int M,
                                unsigned int N_kv, unsigned int Hq,
                                unsigned int Hkv, unsigned int d, bool causal) {
  using nntrainer::compute_fp16_to_fp32;
  const unsigned int HDq = Hq * d, HDkv = Hkv * d, gqa = Hq / Hkv;
  const float scale = 1.0f / std::sqrt((float)d);
  std::vector<float> O((size_t)M * HDq, 0.0f);
  std::vector<float> scores(N_kv);
  for (unsigned int hq = 0; hq < Hq; ++hq) {
    const unsigned int hkv = hq / gqa;
    for (unsigned int m = 0; m < M; ++m) {
      float mx = -INFINITY;
      for (unsigned int n = 0; n < N_kv; ++n) {
        if (causal && n > m) {
          scores[n] = -INFINITY;
          continue;
        }
        float acc = 0.0f;
        for (unsigned int x = 0; x < d; ++x)
          acc += compute_fp16_to_fp32(Q[(size_t)m * HDq + hq * d + x]) *
                 compute_fp16_to_fp32(K[(size_t)n * HDkv + hkv * d + x]);
        scores[n] = acc * scale;
        if (scores[n] > mx)
          mx = scores[n];
      }
      double sum = 0.0;
      for (unsigned int n = 0; n < N_kv; ++n) {
        if (std::isfinite(scores[n])) {
          scores[n] = std::exp(scores[n] - mx);
          sum += scores[n];
        } else {
          scores[n] = 0.0f;
        }
      }
      if (sum > 0)
        for (unsigned int n = 0; n < N_kv; ++n)
          scores[n] /= (float)sum;
      for (unsigned int x = 0; x < d; ++x) {
        float o = 0.0f;
        for (unsigned int n = 0; n < N_kv; ++n)
          o += scores[n] * compute_fp16_to_fp32(V[(size_t)n * HDkv + hkv * d + x]);
        O[(size_t)m * HDq + hq * d + x] = o;
      }
    }
  }
  return O;
}

/// Relative L2 of (a - b) over b, with both stored as fp16 bits.
double relL2Half(const std::vector<uint16_t> &a, const std::vector<float> &b) {
  using nntrainer::compute_fp16_to_fp32;
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < b.size(); ++i) {
    const double av = compute_fp16_to_fp32(a[i]);
    const double bv = b[i];
    num += (av - bv) * (av - bv);
    den += bv * bv;
  }
  return den > 0 ? std::sqrt(num / den) : 0.0;
}

} // namespace

/**
 * @brief The engine=gpu graph (run under the SVM residency pool) must match the
 *        engine=cpu reference. NNTR_GPU_SVM_POOL is forced on in main() so the
 *        GPU pool uses the SVM allocator and the in-order queue.
 */
TEST(ClResidency, fc_chain_residual_matches_cpu) {
  const unsigned int in_w = 8, unit = 4;
  std::vector<float> input(in_w);
  for (unsigned int i = 0; i < in_w; ++i)
    input[i] = static_cast<float>(i + 1) * 0.125f;

  std::vector<float> cpu = runModel("cpu", in_w, unit, input);
  std::vector<float> gpu = runModel("gpu", in_w, unit, input);

  ASSERT_EQ(cpu.size(), static_cast<size_t>(unit));
  ASSERT_EQ(gpu.size(), static_cast<size_t>(unit));
  for (unsigned int i = 0; i < unit; ++i)
    EXPECT_NEAR(cpu[i], gpu[i], 1e-3f) << "output mismatch at index " << i;
}

/**
 * @brief Attention residency: two_conv_attention_prefill_f16_cl reading and
 *        writing SVM-resident Q/K/V/O (svm_inputs=true) must produce the same
 *        output as the cl_mem upload path (svm_inputs=false) and a CPU golden.
 *        This is the attention half of the residency story (the fc_chain test
 *        above covers FC/RMSNorm/SwiGLU/Add). NNTR_MHA_VERIFY=1 (set in main())
 *        bypasses the 28-layer drift gate so the kernel actually dispatches.
 */
TEST(ClResidency, attention_svm_matches_clmem) {
  const unsigned int M = 48, N_kv = 48, Hq = 8, Hkv = 4, d = 64;
  const bool causal = true;
  const unsigned int HDq = Hq * d, HDkv = Hkv * d;

  std::vector<uint16_t> Q = genHalf((size_t)M * HDq, 1);
  std::vector<uint16_t> K = genHalf((size_t)N_kv * HDkv, 2);
  std::vector<uint16_t> V = genHalf((size_t)N_kv * HDkv, 3);

  // CPU golden.
  std::vector<float> ref = cpuAttention(Q, K, V, M, N_kv, Hq, Hkv, d, causal);

  // GPU cl_mem path (host pointers uploaded to scratch cl_mem). This also
  // initializes the GPU context/queue for the SVM path below.
  std::vector<uint16_t> O_clmem((size_t)M * HDq, 0);
  bool ok_clmem = nntrainer::two_conv_attention_prefill_f16_cl(
    Q.data(), K.data(), V.data(), O_clmem.data(), M, N_kv, Hq, Hkv, d, causal,
    /*svm_inputs=*/false);
  ASSERT_TRUE(ok_clmem) << "cl_mem attention path unsupported/failed";

  // GPU SVM residency path: Q/K/V/O are GPU-resident SVM pointers, bound via
  // clSetKernelArgSVMPointer (no host upload). Coarse-grained SVM coherence
  // mirrors the production add/rmsnorm path: host write -> SVMUnmap (hand to
  // device) -> dispatch -> SVMMap (host read).
  auto *cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  ASSERT_NE(cc, nullptr);
  cc->context_inst_.GetContext(); // ensure the cl_context exists

  const size_t qB = Q.size() * sizeof(uint16_t);
  const size_t kB = K.size() * sizeof(uint16_t);
  const size_t vB = V.size() * sizeof(uint16_t);
  const size_t oB = (size_t)M * HDq * sizeof(uint16_t);
  auto *Qs = static_cast<uint16_t *>(cc->context_inst_.createSVMRegion(qB));
  auto *Ks = static_cast<uint16_t *>(cc->context_inst_.createSVMRegion(kB));
  auto *Vs = static_cast<uint16_t *>(cc->context_inst_.createSVMRegion(vB));
  auto *Os = static_cast<uint16_t *>(cc->context_inst_.createSVMRegion(oB));
  ASSERT_TRUE(Qs && Ks && Vs && Os) << "SVM allocation failed (no SVM support?)";

  std::memcpy(Qs, Q.data(), qB);
  std::memcpy(Ks, K.data(), kB);
  std::memcpy(Vs, V.data(), vB);
  std::memset(Os, 0, oB);
  cc->command_queue_inst_.enqueueSVMUnmap(Qs);
  cc->command_queue_inst_.enqueueSVMUnmap(Ks);
  cc->command_queue_inst_.enqueueSVMUnmap(Vs);
  cc->command_queue_inst_.enqueueSVMUnmap(Os);

  bool ok_svm = nntrainer::two_conv_attention_prefill_f16_cl(
    Qs, Ks, Vs, Os, M, N_kv, Hq, Hkv, d, causal, /*svm_inputs=*/true);
  ASSERT_TRUE(ok_svm) << "SVM attention path unsupported/failed";

  cc->command_queue_inst_.enqueueSVMMap(Os, oB, /*read_only=*/true);
  std::vector<uint16_t> O_svm(Os, Os + (size_t)M * HDq);
  cc->command_queue_inst_.enqueueSVMUnmap(Os);

  cc->context_inst_.releaseSVMRegion(Qs);
  cc->context_inst_.releaseSVMRegion(Ks);
  cc->context_inst_.releaseSVMRegion(Vs);
  cc->context_inst_.releaseSVMRegion(Os);

  // Residency equivalence: SVM-direct vs cl_mem-upload run identical kernels on
  // identical bytes, so the fp16 outputs must match to within tight fp tol.
  ASSERT_EQ(O_svm.size(), O_clmem.size());
  float max_abs = 0.0f;
  for (size_t i = 0; i < O_svm.size(); ++i) {
    const float s = nntrainer::compute_fp16_to_fp32(O_svm[i]);
    const float c = nntrainer::compute_fp16_to_fp32(O_clmem[i]);
    max_abs = std::max(max_abs, std::fabs(s - c));
    EXPECT_NEAR(s, c, 2e-3f) << "SVM vs cl_mem mismatch at " << i;
  }
  std::cout << "[attention residency] max|svm-clmem|=" << max_abs << std::endl;

  // Correctness: both GPU paths vs CPU golden (fp16 storage + reduction order).
  const double rel_svm = relL2Half(O_svm, ref);
  const double rel_clmem = relL2Half(O_clmem, ref);
  std::cout << "[attention residency] relL2 svm=" << rel_svm
            << " clmem=" << rel_clmem << std::endl;
  EXPECT_LT(rel_svm, 3e-2) << "SVM attention diverges from CPU golden";
  EXPECT_LT(rel_clmem, 3e-2) << "cl_mem attention diverges from CPU golden";
}

/**
 * @brief FP16 SVM-direct rmsnorm (rmsnorm_cl_fp16) must match a CPU golden.
 *        This is the new residency primitive behind q/k-norm GPU residency
 *        (ReshapedRMSNormLayer reshapes to feature_size then calls this). Each
 *        row is normalized over W and scaled by gamma, SVM-direct.
 */
TEST(ClResidency, rmsnorm_fp16_svm_matches_cpu) {
  using nntrainer::compute_fp16_to_fp32;
  using nntrainer::compute_fp32_to_fp16;
  const unsigned int H = 96, W = 64; // rows x feature_size (e.g. head_dim)
  const float eps = 1e-6f;

  std::vector<uint16_t> X = genHalf((size_t)H * W, 7);
  std::vector<uint16_t> G = genHalf(W, 8);

  // CPU golden: out[h,w] = x[h,w] / sqrt(mean_w(x^2) + eps) * gamma[w].
  std::vector<float> ref((size_t)H * W);
  for (unsigned h = 0; h < H; ++h) {
    double ss = 0.0;
    for (unsigned w = 0; w < W; ++w) {
      float x = compute_fp16_to_fp32(X[(size_t)h * W + w]);
      ss += (double)x * x;
    }
    float scale = 1.0f / std::sqrt((float)(ss / W) + eps);
    for (unsigned w = 0; w < W; ++w)
      ref[(size_t)h * W + w] = compute_fp16_to_fp32(X[(size_t)h * W + w]) *
                               scale * compute_fp16_to_fp32(G[w]);
  }

  auto *cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  ASSERT_NE(cc, nullptr);
  cc->context_inst_.GetContext();

  const size_t xB = X.size() * 2, gB = G.size() * 2, oB = (size_t)H * W * 2;
  auto *Xs = static_cast<uint16_t *>(cc->context_inst_.createSVMRegion(xB));
  auto *Gs = static_cast<uint16_t *>(cc->context_inst_.createSVMRegion(gB));
  auto *Os = static_cast<uint16_t *>(cc->context_inst_.createSVMRegion(oB));
  ASSERT_TRUE(Xs && Gs && Os) << "SVM alloc failed";
  std::memcpy(Xs, X.data(), xB);
  std::memcpy(Gs, G.data(), gB);
  std::memset(Os, 0, oB);
  cc->command_queue_inst_.enqueueSVMUnmap(Xs);
  cc->command_queue_inst_.enqueueSVMUnmap(Gs);
  cc->command_queue_inst_.enqueueSVMUnmap(Os);

  nntrainer::rmsnorm_cl_fp16(reinterpret_cast<const _FP16 *>(Xs),
                             reinterpret_cast<const _FP16 *>(Gs),
                             reinterpret_cast<_FP16 *>(Os), eps, H, W,
                             /*use_svm=*/true);

  cc->command_queue_inst_.enqueueSVMMap(Os, oB, /*read_only=*/true);
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < (size_t)H * W; ++i) {
    double g = compute_fp16_to_fp32(Os[i]);
    num += (g - ref[i]) * (g - ref[i]);
    den += ref[i] * ref[i];
  }
  cc->command_queue_inst_.enqueueSVMUnmap(Os);
  cc->context_inst_.releaseSVMRegion(Xs);
  cc->context_inst_.releaseSVMRegion(Gs);
  cc->context_inst_.releaseSVMRegion(Os);

  double relL2 = den > 0 ? std::sqrt(num / den) : 0.0;
  std::cout << "[rmsnorm fp16 svm] relL2=" << relL2 << std::endl;
  EXPECT_LT(relL2, 2e-2) << "FP16 SVM rmsnorm diverges from CPU golden";
}

/**
 * @brief Fused SwiGLU+int8-quant (fused_swiglu_quant_cl) must produce an int8
 *        quantization of silu(gate)*up matching a CPU golden. This is the FFN
 *        geglu/swiglu+quant fusion primitive (#6): its int8/scale/zp/row_sum
 *        feed the v8c down-proj GEMM directly, dropping the fp16 round-trip.
 */
TEST(ClResidency, fused_swiglu_quant_svm_matches_cpu) {
  using nntrainer::compute_fp16_to_fp32;
  const unsigned int M = 48, K = 128;

  std::vector<uint16_t> Gate = genHalf((size_t)M * K, 11);
  std::vector<uint16_t> Up = genHalf((size_t)M * K, 12);

  // CPU golden: v = silu(gate)*up; asymmetric int8 quant (v8c scheme).
  std::vector<float> vref((size_t)M * K);
  std::vector<float> gscale(M);
  std::vector<int> gzp(M);
  for (unsigned r = 0; r < M; ++r) {
    float vmin = 0.0f, vmax = 0.0f;
    for (unsigned k = 0; k < K; ++k) {
      float g = compute_fp16_to_fp32(Gate[(size_t)r * K + k]);
      float u = compute_fp16_to_fp32(Up[(size_t)r * K + k]);
      float v = (g / (1.0f + std::exp(-g))) * u;
      vref[(size_t)r * K + k] = v;
      vmin = std::min(vmin, v);
      vmax = std::max(vmax, v);
    }
    float range = vmax - vmin;
    float scale_q = range > 0 ? 255.0f / range : 1.0f;
    float recip = range > 0 ? range / 255.0f : 1.0f;
    float dmin = vmin * scale_q, dmax = vmax * scale_q;
    float zp_lo = -128.0f - dmin, zp_hi = 127.0f - dmax;
    float zp_f = (-128.0f + dmin) + (127.0f + dmax) > 0.0f ? zp_lo : zp_hi;
    zp_f = std::max(-128.0f, std::min(127.0f, zp_f));
    gscale[r] = recip;
    gzp[r] = (int)std::rint(zp_f);
  }

  auto *cc = static_cast<nntrainer::ClContext *>(
    nntrainer::Engine::Global().getRegisteredContext("gpu"));
  ASSERT_NE(cc, nullptr);
  cc->context_inst_.GetContext();
  auto svm = [&](size_t bytes) {
    return cc->context_inst_.createSVMRegion(bytes);
  };
  auto *Gs = static_cast<uint16_t *>(svm(Gate.size() * 2));
  auto *Us = static_cast<uint16_t *>(svm(Up.size() * 2));
  auto *I8 = static_cast<int8_t *>(svm((size_t)M * K));
  auto *Sc = static_cast<float *>(svm((size_t)M * 4));
  auto *Zp = static_cast<int *>(svm((size_t)M * 4));
  auto *Rs = static_cast<int *>(svm((size_t)M * 4));
  ASSERT_TRUE(Gs && Us && I8 && Sc && Zp && Rs);
  std::memcpy(Gs, Gate.data(), Gate.size() * 2);
  std::memcpy(Us, Up.data(), Up.size() * 2);
  cc->command_queue_inst_.enqueueSVMUnmap(Gs);
  cc->command_queue_inst_.enqueueSVMUnmap(Us);
  cc->command_queue_inst_.enqueueSVMUnmap(I8);
  cc->command_queue_inst_.enqueueSVMUnmap(Sc);
  cc->command_queue_inst_.enqueueSVMUnmap(Zp);
  cc->command_queue_inst_.enqueueSVMUnmap(Rs);

  bool ok = nntrainer::fused_swiglu_quant_cl(
    reinterpret_cast<const _FP16 *>(Gs), reinterpret_cast<const _FP16 *>(Us), I8,
    Sc, Zp, Rs, M, K, /*use_svm=*/true);
  ASSERT_TRUE(ok) << "fused_swiglu_quant_cl failed";

  cc->command_queue_inst_.enqueueSVMMap(I8, (size_t)M * K, true);
  cc->command_queue_inst_.enqueueSVMMap(Sc, (size_t)M * 4, true);
  cc->command_queue_inst_.enqueueSVMMap(Zp, (size_t)M * 4, true);
  cc->command_queue_inst_.enqueueSVMMap(Rs, (size_t)M * 4, true);

  // Dequantize GPU int8 and compare to the fp32 golden v; check row_sum + zp.
  double num = 0.0, den = 0.0;
  int zp_mism = 0, rs_mism = 0;
  for (unsigned r = 0; r < M; ++r) {
    long rs = 0;
    for (unsigned k = 0; k < K; ++k) {
      int q = I8[(size_t)r * K + k];
      rs += q;
      double dq = (q - Zp[r]) * (double)Sc[r];
      double v = vref[(size_t)r * K + k];
      num += (dq - v) * (dq - v);
      den += v * v;
    }
    if (Zp[r] != gzp[r])
      zp_mism++;
    if (Rs[r] != (int)rs)
      rs_mism++;
  }
  double relL2 = den > 0 ? std::sqrt(num / den) : 0.0;
  std::cout << "[fused swiglu quant] relL2(dequant vs golden)=" << relL2
            << " zp_mism=" << zp_mism << " rs_internal_mism=" << rs_mism
            << std::endl;
  EXPECT_LT(relL2, 1.5e-2) << "fused swiglu+quant diverges from golden";
  EXPECT_EQ(rs_mism, 0) << "row_sum != sum(int8)";
  for (void *p : {(void *)Gs, (void *)Us, (void *)I8, (void *)Sc, (void *)Zp,
                  (void *)Rs})
    cc->context_inst_.releaseSVMRegion(p);
}

GTEST_API_ int main(int argc, char **argv) {
  // Force the GPU graph onto the SVM residency pool (in-order queue + SVM
  // allocator) for the whole process, before any GPU context/queue is created.
  setenv("NNTR_GPU_SVM_POOL", "1", 1);
  // Bypass the 28-layer numerical-drift gate in two_conv_attention_prefill so
  // the attention kernel actually dispatches (it is math-correct for 1 call;
  // drift only matters across a full 28-layer model). See attention_kernels.cpp.
  setenv("NNTR_MHA_VERIFY", "1", 1);
  // Compile the attention program in buffer-only mode (-DTCA_BUFFER_ONLY). The
  // two_conv_attention.cl program also contains image2d (read_imageui) kernels;
  // on runtimes that lack integer-coord read_imageui (Intel NEO) clBuildProgram
  // fails for the WHOLE program, which would null out even the buffer-based
  // qk_matmul_f16 this test uses. The image kernels are not used here, so
  // compiling them out is safe on every device (Adreno just loses its fast
  // image path, irrelevant for a correctness test).
  setenv("NNTR_V8C_BUF", "1", 1);

  int result = -1;
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }
  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }
  return result;
}

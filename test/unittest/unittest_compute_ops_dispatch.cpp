// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   unittest_compute_ops_dispatch.cpp
 * @date   25 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Verify that a per-Context ComputeOps installed via ContextData
 *         actually reaches Tensor::dot / multiply / add through the
 *         tensor's attached ContextData. End-to-end check that
 *         vendor backend dispatch works with virtual dispatch (no
 *         preprocessor branches at the call site).
 */

#include <common_properties.h>
#include <compute_ops.h>
#include <context_data.h>
#include <gtest/gtest.h>
#include <tensor.h>

#include <atomic>
#include <cmath>
#include <memory>

namespace {

/**
 * @brief Per-test counters incremented by the mock subclass.
 *
 * The mock forwards each op to the real (CPU) backend so that result
 * correctness is preserved; the counters confirm dispatch reached the
 * mock and not the global singleton directly.
 */
struct CallCounters {
  std::atomic<int> sgemm{0};
  std::atomic<int> sgemv{0};
  std::atomic<int> ele_mul{0};
  std::atomic<int> ele_add{0};
  std::atomic<int> scopy{0};
  std::atomic<int> rms_reverse_norm{0};
  std::atomic<int> scalar_mul{0};
  std::atomic<int> softcap{0};
  std::atomic<int> rms_norm{0};
};

/**
 * @brief Mock ComputeOps subclass: forwards to a "real" backend
 *        (the global one) for correctness while bumping per-op counters.
 *
 * Because every base-class default just throws, this only overrides
 * the ops the tests exercise. Anything the test setup happens to
 * trigger that's not overridden here would throw — the tests stay
 * inside the overridden subset (sgemm/sgemv/ele_mul/ele_add/scopy).
 */
class MockComputeOps : public nntrainer::ComputeOps {
public:
  MockComputeOps(nntrainer::ComputeOps *real, CallCounters *c) :
    real_(real), counters_(c) {}

  void sgemm_fp32(unsigned int o, bool tA, bool tB, unsigned int M,
                  unsigned int N, unsigned int K, float a, const float *A,
                  unsigned int lda, const float *B, unsigned int ldb, float b,
                  float *C, unsigned int ldc) override {
    counters_->sgemm++;
    real_->sgemm_fp32(o, tA, tB, M, N, K, a, A, lda, B, ldb, b, C, ldc);
  }
  void sgemv_fp32(unsigned int o, bool tA, unsigned int M, unsigned int N,
                  float a, const float *A, unsigned int lda, const float *X,
                  unsigned int iX, float b, float *Y,
                  unsigned int iY) override {
    counters_->sgemv++;
    real_->sgemv_fp32(o, tA, M, N, a, A, lda, X, iX, b, Y, iY);
  }
  void ele_mul_fp32(unsigned int N, const float *X, const float *Y, float *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    counters_->ele_mul++;
    real_->ele_mul_fp32(N, X, Y, Z, a, b, is, os);
  }
  void ele_add_fp32(unsigned int N, const float *X, const float *Y, float *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    counters_->ele_add++;
    real_->ele_add_fp32(N, X, Y, Z, a, b, is, os);
  }
  void scopy_fp32(unsigned int N, const float *X, unsigned int iX, float *Y,
                  unsigned int iY) override {
    counters_->scopy++;
    real_->scopy_fp32(N, X, iX, Y, iY);
  }
  void rms_reverse_norm(nntrainer::Tensor &in, nntrainer::Tensor &out,
                        const nntrainer::Tensor &weight,
                        const nntrainer::Tensor &out_scale, float epsilon,
                        unsigned int active_rows,
                        unsigned int row_offset) override {
    counters_->rms_reverse_norm++;
    real_->rms_reverse_norm(in, out, weight, out_scale, epsilon, active_rows,
                            row_offset);
  }
  void scalar_mul(const nntrainer::Tensor &in, nntrainer::Tensor &out,
                  float scale) override {
    counters_->scalar_mul++;
    real_->scalar_mul(in, out, scale);
  }
  void softcap(const nntrainer::Tensor &in, nntrainer::Tensor &out, float cap,
               int act_type) override {
    counters_->softcap++;
    real_->softcap(in, out, cap, act_type);
  }
  void rms_norm(const nntrainer::Tensor &in, nntrainer::Tensor &out,
                const nntrainer::Tensor &gamma, float epsilon,
                unsigned int active_rows, unsigned int row_offset) override {
    counters_->rms_norm++;
    real_->rms_norm(in, out, gamma, epsilon, active_rows, row_offset);
  }

private:
  nntrainer::ComputeOps *real_;
  CallCounters *counters_;
};

class ComputeOpsDispatchTest : public ::testing::Test {
protected:
  void SetUp() override {
    counters = std::make_unique<CallCounters>();
    nntrainer::ensureComputeOps();
    mock_ops = std::make_unique<MockComputeOps>(nntrainer::getComputeOps(),
                                                counters.get());
    ct_data = std::make_shared<nntrainer::ContextData>();
    ct_data->setComputeOps(mock_ops.get());
  }

  std::unique_ptr<CallCounters> counters;
  std::unique_ptr<MockComputeOps> mock_ops;
  std::shared_ptr<nntrainer::ContextData> ct_data;
};

} // namespace

/**
 * @brief A Tensor with no attached ContextData should fall back to the
 *        global ops; no mock counter increments.
 */
TEST_F(ComputeOpsDispatchTest, FallbackToGlobalWhenNoContextData) {
  nntrainer::Tensor a(1, 1, 4, 4);
  nntrainer::Tensor b(1, 1, 4, 4);
  a.setValue(1.0f);
  b.setValue(1.0f);

  auto out = a.dot(b);

  EXPECT_EQ(counters->sgemm.load(), 0);
  EXPECT_EQ(counters->sgemv.load(), 0);
}

/**
 * @brief When a ContextData with the mock subclass is attached to a
 *        Tensor, calling .dot dispatches through the mock.
 */
TEST_F(ComputeOpsDispatchTest, DotDispatchesThroughAttachedContextOps) {
  nntrainer::Tensor a(1, 1, 4, 4);
  nntrainer::Tensor b(1, 1, 4, 4);
  a.setValue(1.0f);
  b.setValue(1.0f);

  a.setContextData(ct_data);
  auto out = a.dot(b);

  EXPECT_GT(counters->sgemm.load() + counters->sgemv.load(), 0);
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 0, 0), 4.0f);
}

/**
 * @brief Element-wise multiply through the attached ContextData should
 *        invoke the mock's ele_mul_fp32.
 */
TEST_F(ComputeOpsDispatchTest, MultiplyDispatchesThroughAttachedContextOps) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  a.setContextData(ct_data);
  a.multiply(b, out);

  EXPECT_GT(counters->ele_mul.load(), 0);
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 0, 0), 6.0f);
}

/**
 * @brief Element-wise add through the attached ContextData should
 *        invoke the mock's ele_add_fp32.
 */
TEST_F(ComputeOpsDispatchTest, AddDispatchesThroughAttachedContextOps) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  a.setContextData(ct_data);
  a.add(b, out);

  EXPECT_GT(counters->ele_add.load(), 0);
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 0, 0), 5.0f);
}

/**
 * @brief Result tensor of a binary op inherits ContextData from `this`,
 *        so a chained op on the result keeps dispatching through the
 *        same backend.
 */
TEST_F(ComputeOpsDispatchTest, ResultInheritsContextDataFromOperand) {
  nntrainer::Tensor a(1, 1, 4, 4);
  nntrainer::Tensor b(1, 1, 4, 4);
  a.setValue(1.0f);
  b.setValue(1.0f);

  a.setContextData(ct_data);

  auto first = a.dot(b);
  EXPECT_EQ(first.getContextData().get(), ct_data.get());

  int before = counters->sgemm.load() + counters->sgemv.load();
  auto second = first.dot(b);
  int after = counters->sgemm.load() + counters->sgemv.load();
  EXPECT_GT(after, before);
}

/**
 * @brief Replacing the attached ContextData with a different subclass
 *        rebinds dispatch on subsequent calls — the runtime swap
 *        property required for hot-swapping vendor contexts.
 */
TEST_F(ComputeOpsDispatchTest, SwappingContextDataRebindsDispatch) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out1(1, 1, 1, 8);
  nntrainer::Tensor out2(1, 1, 1, 8);

  a.setContextData(ct_data);
  a.multiply(b, out1);
  int after_first = counters->ele_mul.load();
  EXPECT_GT(after_first, 0);

  // Swap to a fresh ContextData carrying its own MockComputeOps with
  // independent counters. Subsequent ops bump fresh counters only.
  auto fresh_counters = std::make_unique<CallCounters>();
  auto fresh_mock = std::make_unique<MockComputeOps>(nntrainer::getComputeOps(),
                                                     fresh_counters.get());
  auto fresh_ct = std::make_shared<nntrainer::ContextData>();
  fresh_ct->setComputeOps(fresh_mock.get());
  a.setContextData(fresh_ct);

  a.multiply(b, out2);
  EXPECT_EQ(counters->ele_mul.load(), after_first);
  EXPECT_GT(fresh_counters->ele_mul.load(), 0);
}

/**
 * @brief Cross-vendor mismatch — when two operands of a binary op
 *        carry DIFFERENT ContextData (e.g. one CPU-resident tensor,
 *        one OpenCL-resident tensor), the op must throw rather than
 *        silently dispatch through one side's ops onto the other
 *        side's incompatible memory. This is the assertion that
 *        protects against the most insidious mixed-backend bug.
 */
TEST_F(ComputeOpsDispatchTest, BinaryOpThrowsOnContextMismatch) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  // Two distinct ContextData instances simulate two different vendor
  // contexts (e.g. CPU + OpenCL) — same kind of mock here, but the
  // identity of the ContextData pointers differs.
  auto ct_other = std::make_shared<nntrainer::ContextData>();
  ct_other->setComputeOps(mock_ops.get());

  a.setContextData(ct_data);
  b.setContextData(ct_other);

  EXPECT_THROW(a.multiply(b, out), std::invalid_argument);
  EXPECT_THROW(a.add(b, out), std::invalid_argument);
  EXPECT_THROW(a.divide(b, out), std::invalid_argument);
}

/**
 * @brief Same ContextData identity on both operands → no throw.
 *        Confirms the mismatch check is keyed on identity, not on
 *        nullness alone (which would be backward-compatibility break).
 */
TEST_F(ComputeOpsDispatchTest, BinaryOpAcceptsSameContext) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  a.setContextData(ct_data);
  b.setContextData(ct_data); // same instance, not a copy

  EXPECT_NO_THROW(a.multiply(b, out));
}

/**
 * @brief One operand has no ContextData → permissive (legacy code
 *        path). A tensor created without ever touching ContextData
 *        falls back to the global ops table; binary-op'ing it with
 *        a context-attached tensor must NOT throw — that would break
 *        every existing test and call site.
 */
TEST_F(ComputeOpsDispatchTest, BinaryOpAcceptsOneSideUnattached) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  a.setContextData(ct_data);
  // b has no ContextData — legacy code path

  EXPECT_NO_THROW(a.multiply(b, out));
}

/**
 * @brief Tensor::to(target_ct) deep-copies and re-tags. Result owns
 *        the new ContextData; original is unchanged. After to(), a
 *        previously-mismatched binary op on the migrated tensor
 *        succeeds.
 */
TEST_F(ComputeOpsDispatchTest, ToMigratesContextDataAndUnblocksOp) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  auto ct_other = std::make_shared<nntrainer::ContextData>();
  ct_other->setComputeOps(mock_ops.get());

  a.setContextData(ct_data);
  b.setContextData(ct_other);

  // Migrate b onto a's context. Original b stays on ct_other.
  nntrainer::Tensor b_migrated = b.to(ct_data);
  EXPECT_EQ(b_migrated.getContextData().get(), ct_data.get());
  EXPECT_EQ(b.getContextData().get(), ct_other.get()); // unchanged

  // Now a.multiply(b_migrated) is on the same context — no throw.
  EXPECT_NO_THROW(a.multiply(b_migrated, out));
}

/**
 * @brief The reverse-RMSNorm whole-op (PLE post_norm) dispatches
 *        through the attached ContextData ops — the dispatch seam the layer's
 *        former open-coded body structurally could not test — and the
 *        (active_rows, row_offset) window is honoured: rows outside the
 *        window are untouched.
 */
TEST_F(ComputeOpsDispatchTest,
       RmsReverseNormDispatchesThroughAttachedContextOps) {
  nntrainer::Tensor in(1, 1, 2, 4);
  nntrainer::Tensor out(1, 1, 2, 4);
  nntrainer::Tensor weight(1, 1, 1, 4);
  nntrainer::Tensor out_scale(1, 1, 1, 1);
  in.setValue(1.0f);
  out.setValue(0.0f);
  weight.setValue(2.0f);
  out_scale.setValue(3.0f);

  in.setContextData(ct_data);
  in.getOps()->rms_reverse_norm(in, out, weight, out_scale, /*epsilon=*/0.0f,
                                /*active_rows=*/1, /*row_offset=*/0);

  EXPECT_GT(counters->rms_reverse_norm.load(), 0);
  // out = out_scale * (x*w) * rsqrt(mean((x*w)^2) + eps)
  //     = 3 * (1*2) * rsqrt(4) = 3, exactly, in fp32.
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 0, 0), 3.0f);
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 0, 3), 3.0f);
  // Row 1 sits outside active_rows=1 and must be untouched.
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 1, 0), 0.0f);
}

/**
 * @brief The scalar-multiply whole-op dispatches through the attached
 *        ContextData ops — the dispatch seam the layer's former open-coded
 *        body structurally could not test — and the Cpu impl computes the
 *        exact product.
 */
TEST_F(ComputeOpsDispatchTest, ScalarMulDispatchesThroughAttachedContextOps) {
  nntrainer::Tensor in(1, 1, 2, 4);
  nntrainer::Tensor out(1, 1, 2, 4);
  in.setValue(2.0f);
  out.setValue(0.0f);

  in.setContextData(ct_data);
  in.getOps()->scalar_mul(in, out, 3.0f);

  EXPECT_GT(counters->scalar_mul.load(), 0);
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 0, 0), 6.0f);
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 1, 3), 6.0f);
}

/**
 * @brief The logit-softcapping whole-op dispatches through the attached
 *        ContextData ops, and the Cpu impl computes cap * tanh(in / cap)
 *        (tanh is the activation every shipped configuration sets).
 */
TEST_F(ComputeOpsDispatchTest, SoftcapDispatchesThroughAttachedContextOps) {
  nntrainer::Tensor in(1, 1, 1, 8);
  nntrainer::Tensor out(1, 1, 1, 8);
  in.setValue(1.0f);
  out.setValue(0.0f);

  in.setContextData(ct_data);
  in.getOps()->softcap(in, out, /*cap=*/2.0f,
                       static_cast<int>(nntrainer::ActivationType::ACT_TANH));

  EXPECT_GT(counters->softcap.load(), 0);
  // out = cap * tanh(in / cap) = 2 * tanh(0.5). The ActiFunc tanh is the
  // algebraically-identical 2*sigmoid(2x)-1 form, so allow a few float ULPs.
  EXPECT_NEAR(out.getValue<float>(0, 0, 0, 0), 2.0f * std::tanh(0.5f), 1e-5f);
  EXPECT_NEAR(out.getValue<float>(0, 0, 0, 7), 2.0f * std::tanh(0.5f), 1e-5f);
}

/**
 * @brief The RMS-normalization whole-op dispatches through the attached
 *        ContextData ops, the Cpu impl computes the analytic value, and the
 *        (active_rows, row_offset) window is honored — rows outside the
 *        window stay untouched.
 */
TEST_F(ComputeOpsDispatchTest, RmsNormDispatchesThroughAttachedContextOps) {
  nntrainer::Tensor in(1, 1, 3, 4);
  nntrainer::Tensor out(1, 1, 3, 4);
  nntrainer::Tensor gamma(1, 1, 1, 4);
  in.setValue(2.0f);
  gamma.setValue(2.0f);
  out.setValue(0.0f);

  in.setContextData(ct_data);
  // Window: normalize only the first 2 of 3 rows.
  in.getOps()->rms_norm(in, out, gamma, /*epsilon=*/0.0f, /*active_rows=*/2,
                        /*row_offset=*/0);

  EXPECT_GT(counters->rms_norm.load(), 0);
  // Constant row: rms = sqrt(mean(2^2)) = 2, so each element normalizes to
  // 1.0; gamma = 2 scales it to 2.0 (all powers of two — exact in fp32 on
  // every arch path).
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 0, 0), 2.0f);
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 1, 3), 2.0f);
  // Row 2 is outside the active window: still the 0.0 sentinel.
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 2, 0), 0.0f);
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 2, 3), 0.0f);
}

/**
 * @brief The reverse-RMSNorm whole-op (PLE post_norm) dispatches
 *        through the attached ContextData ops — the dispatch seam the layer's
 *        former open-coded body structurally could not test — and the
 *        (active_rows, row_offset) window is honoured: rows outside the
 *        window are untouched.
 */
TEST_F(ComputeOpsDispatchTest,
       RmsReverseNormDispatchesThroughAttachedContextOps) {
  nntrainer::Tensor in(1, 1, 2, 4);
  nntrainer::Tensor out(1, 1, 2, 4);
  nntrainer::Tensor weight(1, 1, 1, 4);
  nntrainer::Tensor out_scale(1, 1, 1, 1);
  in.setValue(1.0f);
  out.setValue(0.0f);
  weight.setValue(2.0f);
  out_scale.setValue(3.0f);

  in.setContextData(ct_data);
  in.getOps()->rms_reverse_norm(in, out, weight, out_scale, /*epsilon=*/0.0f,
                                /*active_rows=*/1, /*row_offset=*/0);

  EXPECT_GT(counters->rms_reverse_norm.load(), 0);
  // out = out_scale * (x*w) * rsqrt(mean((x*w)^2) + eps)
  //     = 3 * (1*2) * rsqrt(4) = 3, exactly, in fp32.
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 0, 0), 3.0f);
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 0, 3), 3.0f);
  // Row 1 sits outside active_rows=1 and must be untouched.
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 1, 0), 0.0f);
}

/* ==========================================================================
 * CUDA whole-op table completeness
 *
 * CudaComputeOps inherits CpuComputeOps, so a whole-op the CUDA table does NOT
 * override silently runs the inherited HOST body. On a discrete GPU the CUDA
 * context arms the device-only activation pool (NNTR_CUDA_DEV_ACT, set by the
 * CudaContext constructor), where an activation is cudaMalloc memory the CPU
 * cannot address -- so that inherited host body faults, several frames deep
 * inside an AVX2 intrinsic, with nothing naming the op.
 *
 * INVARIANT: every whole-op declared on the ComputeOps base has a
 * CudaComputeOps override (a device kernel, or a named guard, or both).
 *
 * The two tests below are complementary and BOTH are needed:
 *   - HasCudaOverride  checks the ops this file knows about, at compile time.
 *     It cannot see an op added to the base after this file was written.
 *   - CensusCoversHeader  re-reads compute_ops.h and fails on any whole-op in
 *     it that this file does not know about. That is the half that catches a
 *     lane adding a whole-op without a CUDA override -- the way this gap was
 *     opened in the first place.
 *
 * TWO sections are censused, because the invariant has nothing to do with
 * which banner an op sits under:
 *   1. the whole-op (Tensor-level) section, and
 *   2. the raw-pointer COPY family (scopy_* / copy_*), which is spread across
 *      the "FP32 BLAS", "FP32 Data conversion / Copy", "Data conversion
 *      (int8 -> FP32)", "FP16 BLAS" and "FP16 Data conversion" banners.
 * Anchoring only on the whole-op banner left the entire copy family
 * structurally outside the enforcement: `Tensor::copy` on a QINT8 activation
 * dispatches ComputeOps::scopy_s8, which had no CUDA override and no row here,
 * so nothing could have flagged it.
 * ========================================================================== */
#ifdef ENABLE_CUDA

#include <cuda_compute_ops.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

/**
 * @brief Deduce the class a member function was DECLARED in.
 *
 * The type of `&Derived::m` is "pointer to member of B" where B is the class
 * that DECLARES m -- not the class named in the qualified-id. So when
 * CudaComputeOps overrides an op, `&CudaComputeOps::op` is a
 * `CudaComputeOps::*`; when it does not, name lookup finds the inherited
 * declaration and the same expression is a `CpuComputeOps::*` (or
 * `ComputeOps::*`). That difference is the override check, resolved entirely
 * at compile time and without depending on vtable layout.
 *
 * Declared, never defined: used only inside decltype.
 */
template <typename R, typename C, typename... A>
C *declaring_class(R (C::*)(A...));
template <typename R, typename C, typename... A>
C *declaring_class(R (C::*)(A...) const);

/** @brief true iff CudaComputeOps declares its own @p op */
#define CUDA_DECLARES(op)                                                      \
  (std::is_same<decltype(declaring_class(&nntrainer::CudaComputeOps::op)),      \
                nntrainer::CudaComputeOps *>::value)

struct WholeOp {
  const char *name;
  bool cuda_override;
};

/**
 * @brief The whole-op census. One row per whole-op on the ComputeOps base.
 *
 * Adding a row is not optional bookkeeping: CensusCoversHeader below fails if
 * compute_ops.h declares a whole-op that has no row here.
 */
const WholeOp kWholeOps[] = {
  {"geglu", CUDA_DECLARES(geglu)},
  {"swiglu", CUDA_DECLARES(swiglu)},
  {"sigmoid_glu", CUDA_DECLARES(sigmoid_glu)},
  {"sigmoid_add", CUDA_DECLARES(sigmoid_add)},
  {"rms_reverse_norm", CUDA_DECLARES(rms_reverse_norm)},
  {"residual_op", CUDA_DECLARES(residual_op)},
  {"fc", CUDA_DECLARES(fc)},
  {"fc_prebuild_weight", CUDA_DECLARES(fc_prebuild_weight)},
  {"apply_activation", CUDA_DECLARES(apply_activation)},
  {"scalar_mul", CUDA_DECLARES(scalar_mul)},
  {"softcap", CUDA_DECLARES(softcap)},
  {"rms_norm", CUDA_DECLARES(rms_norm)},
};

/**
 * @brief Placeholder for a row whose op does not exist in THIS build.
 *
 * The census reads compute_ops.h as TEXT, so it sees the ops declared inside
 * `#ifdef ENABLE_FP16` whether or not this translation unit has FP16. The row
 * must therefore exist unconditionally, while CUDA_DECLARES() on it can only
 * be compiled when the member does. When it does not, there is no op to
 * override and nothing to enforce -- the row is name-only.
 */
constexpr bool kOpAbsentInThisBuild = true;

/**
 * @brief The copy-family census. One row per scopy_* / copy_* op on the base.
 *
 * Same invariant as kWholeOps, one abstraction level down: these take raw
 * pointers rather than Tensors, and an op with no CudaComputeOps override
 * inherits a CpuComputeOps host element loop that dereferences X and Y --
 * which on the device-only activation pool is exactly the fault this table
 * exists to prevent. An override may be a device implementation (a
 * stream-ordered cudaMemcpyAsync for the byte-identical moves, staging for the
 * fp32<->fp16 converters) or a named refusal; it may not be absent.
 */
const WholeOp kCopyOps[] = {
  {"scopy_fp32", CUDA_DECLARES(scopy_fp32)},
  {"scopy_u8", CUDA_DECLARES(scopy_u8)},
  {"scopy_s8", CUDA_DECLARES(scopy_s8)},
  {"scopy_int4_to_float32", CUDA_DECLARES(scopy_int4_to_float32)},
  {"copy_s16_fp32", CUDA_DECLARES(copy_s16_fp32)},
  {"copy_u16_fp32", CUDA_DECLARES(copy_u16_fp32)},
  {"copy_fp32_u32", CUDA_DECLARES(copy_fp32_u32)},
  {"copy_fp32_u16", CUDA_DECLARES(copy_fp32_u16)},
  {"copy_fp32_u8", CUDA_DECLARES(copy_fp32_u8)},
  {"copy_fp32_s16", CUDA_DECLARES(copy_fp32_s16)},
  {"copy_fp32_s8", CUDA_DECLARES(copy_fp32_s8)},
  {"scopy_int8_to_fp32_u", CUDA_DECLARES(scopy_int8_to_fp32_u)},
  {"scopy_int8_to_fp32_s", CUDA_DECLARES(scopy_int8_to_fp32_s)},
#ifdef ENABLE_FP16
  {"scopy_fp16", CUDA_DECLARES(scopy_fp16)},
  {"scopy_fp32_to_fp16", CUDA_DECLARES(scopy_fp32_to_fp16)},
  {"scopy_fp16_to_fp32", CUDA_DECLARES(scopy_fp16_to_fp32)},
  {"scopy_int4_to_float16", CUDA_DECLARES(scopy_int4_to_float16)},
  {"scopy_int8_to_float16_u", CUDA_DECLARES(scopy_int8_to_float16_u)},
  {"scopy_int8_to_float16_s", CUDA_DECLARES(scopy_int8_to_float16_s)},
#else
  {"scopy_fp16", kOpAbsentInThisBuild},
  {"scopy_fp32_to_fp16", kOpAbsentInThisBuild},
  {"scopy_fp16_to_fp32", kOpAbsentInThisBuild},
  {"scopy_int4_to_float16", kOpAbsentInThisBuild},
  {"scopy_int8_to_float16_u", kOpAbsentInThisBuild},
  {"scopy_int8_to_float16_s", kOpAbsentInThisBuild},
#endif
};

/**
 * @brief Is @p name a member of the copy family?
 *
 * Purely lexical, on purpose: the section this predicate filters is the whole
 * raw-pointer half of the class, so a NEW copy op lands in the census by
 * virtue of being named like one, wherever its author files it. (Every
 * scopy_* / copy_* on the base today is a copy or a representation conversion;
 * nothing else in that half carries either prefix.)
 */
bool isCopyOp(const std::string &name) {
  return name.rfind("scopy_", 0) == 0 || name.rfind("copy_", 0) == 0;
}

/** @brief Every declaration in the section belongs to this census. */
bool acceptAll(const std::string &) { return true; }

/**
 * @brief Ops exempt from the override requirement.
 *
 * Only the supports_*() capability predicates: they carry a safe default, take
 * no Tensor, and run no math -- so they cannot dereference a device pointer.
 * Any other exemption has to be argued in code, here.
 */
bool isExempt(const std::string &name) {
  return name.rfind("supports_", 0) == 0;
}

/** @brief Strip // and block comments so they cannot be parsed as code. */
std::string stripComments(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    if (s[i] == '/' && i + 1 < s.size() && s[i + 1] == '/') {
      while (i < s.size() && s[i] != '\n')
        ++i;
    } else if (s[i] == '/' && i + 1 < s.size() && s[i + 1] == '*') {
      i += 2;
      while (i + 1 < s.size() && !(s[i] == '*' && s[i + 1] == '/'))
        ++i;
      i = std::min(i + 2, s.size());
    } else {
      out.push_back(s[i++]);
    }
  }
  return out;
}

/**
 * @brief Names of every `virtual ... name(` declaration in @p block.
 *
 * The name is the LAST identifier before the argument list, which is true for
 * any spelling of a C++ declarator this header uses (and survives the
 * declaration being wrapped across lines, since the block is scanned as one
 * string).
 */
std::vector<std::string> declaredVirtuals(const std::string &block) {
  std::vector<std::string> names;
  const std::string kw = "virtual";
  size_t pos = 0;
  while ((pos = block.find(kw, pos)) != std::string::npos) {
    const size_t after = pos + kw.size();
    const bool word_start =
      (pos == 0) || !(std::isalnum((unsigned char)block[pos - 1]) ||
                      block[pos - 1] == '_');
    const bool word_end = after < block.size() &&
                          !(std::isalnum((unsigned char)block[after]) ||
                            block[after] == '_');
    const size_t open = block.find('(', after);
    if (!word_start || !word_end || open == std::string::npos) {
      pos = after;
      continue;
    }
    std::string decl = block.substr(after, open - after);
    size_t e = decl.size();
    while (e > 0 && !(std::isalnum((unsigned char)decl[e - 1]) ||
                      decl[e - 1] == '_'))
      --e;
    size_t b = e;
    while (b > 0 &&
           (std::isalnum((unsigned char)decl[b - 1]) || decl[b - 1] == '_'))
      --b;
    if (e > b)
      names.push_back(decl.substr(b, e - b));
    pos = open;
  }
  return names;
}

/**
 * @brief Reconcile the virtuals parsed out of a header section against a
 *        census array, in BOTH directions.
 *
 * Forward: a declaration with no row means nothing checks it. Reverse: a row
 * with no declaration is a stale census entry pointing at an op that moved or
 * was deleted, which would otherwise keep passing forever.
 *
 * @param declared   names parsed from the header section
 * @param known      names the census array carries
 * @param census     census array's identifier, for the failure message
 * @param accept     which declared names this census is responsible for
 */
void reconcile(const std::vector<std::string> &declared,
               const std::set<std::string> &known, const char *census,
               bool (*accept)(const std::string &)) {
  std::set<std::string> seen;
  for (const auto &name : declared) {
    if (isExempt(name) || !accept(name))
      continue;
    seen.insert(name);
    EXPECT_EQ(known.count(name), 1u)
      << "ComputeOps::" << name << " is declared on the base with NO row in "
      << census
      << ", so nothing checks whether CudaComputeOps overrides it. Add the row"
         " (and the CudaComputeOps override it will then require).";
  }
  for (const auto &name : known)
    EXPECT_EQ(seen.count(name), 1u)
      << census << " lists '" << name
      << "' but compute_ops.h no longer declares it in the section this census"
         " covers -- stale census row.";
}

} // namespace

/**
 * @brief Every whole-op this file knows about is overridden by CudaComputeOps.
 *
 * A row that is false means the op falls through to the inherited CpuComputeOps
 * body: host math on what may be a device-only pointer.
 */
TEST(CudaWholeOpTable, HasCudaOverride) {
  for (const auto &op : kWholeOps) {
    EXPECT_TRUE(op.cuda_override)
      << "ComputeOps::" << op.name
      << " has NO CudaComputeOps override -- it inherits the CpuComputeOps host"
         " body, which dereferences operands the CPU cannot address when the"
         " device-only activation pool (NNTR_CUDA_DEV_ACT) is armed. Add a"
         " device implementation or a named guard in"
         " nntrainer/cuda/cuda_compute_ops.{h,cpp}.";
  }
}

/**
 * @brief The census above covers EVERY whole-op the base actually declares.
 *
 * Re-reads compute_ops.h so that adding a whole-op to the base without giving
 * CudaComputeOps an override (and without adding a row above) fails here
 * instead of surfacing as a SIGSEGV in a CUDA decode months later.
 */
TEST(CudaWholeOpTable, CensusCoversHeader) {
#ifndef NNTR_COMPUTE_OPS_HEADER
  GTEST_SKIP() << "NNTR_COMPUTE_OPS_HEADER not defined by the build";
#else
  std::ifstream in(NNTR_COMPUTE_OPS_HEADER);
  if (!in) {
    GTEST_SKIP() << "compute_ops.h not readable at " << NNTR_COMPUTE_OPS_HEADER
                 << " (installed test run); the compile-time half still ran";
  }
  std::ostringstream buf;
  buf << in.rdbuf();
  const std::string src = buf.str();

  // The whole-op section runs from its banner to the class' `protected:`.
  // Located on the RAW text: the banner is itself a comment, so the section
  // must be carved out before comments are stripped from its body.
  const size_t begin = src.find("Whole-op (Tensor-level) ops");
  ASSERT_NE(begin, std::string::npos)
    << "the whole-op section banner moved in compute_ops.h; this census can no"
       " longer find the ops it must cover -- re-anchor it.";
  const size_t end = src.find("protected:", begin);
  ASSERT_NE(end, std::string::npos)
    << "no `protected:` after the whole-op banner in compute_ops.h; re-anchor.";

  const std::vector<std::string> declared =
    declaredVirtuals(stripComments(src.substr(begin, end - begin)));
  ASSERT_FALSE(declared.empty())
    << "parsed zero virtuals out of the whole-op section -- the parser, not the"
       " table, is broken.";

  std::set<std::string> known;
  for (const auto &op : kWholeOps)
    known.insert(op.name);

  reconcile(declared, known, "kWholeOps", acceptAll);
#endif
}

/**
 * @brief Every copy op this file knows about is overridden by CudaComputeOps.
 *
 * Exactly the whole-op requirement, applied to the raw-pointer copy family. A
 * false row means the op inherits the CpuComputeOps element loop, which reads
 * X and writes Y on the host -- undefined on a cudaMalloc pointer.
 */
TEST(CudaCopyOpTable, HasCudaOverride) {
  for (const auto &op : kCopyOps) {
    EXPECT_TRUE(op.cuda_override)
      << "ComputeOps::" << op.name
      << " has NO CudaComputeOps override -- it inherits the CpuComputeOps host"
         " copy loop, which dereferences the source and destination the CPU"
         " cannot address when the device-only activation pool"
         " (NNTR_CUDA_DEV_ACT) is armed. Add a device implementation or a named"
         " refusal in nntrainer/cuda/cuda_compute_ops.{h,cpp}.";
  }
}

/**
 * @brief kCopyOps covers EVERY copy op the base actually declares.
 *
 * The second anchored section. The whole-op census stops at the class'
 * `protected:` AFTER the whole-op banner, so it reaches none of the copy
 * family; this one takes the complementary region -- the class declaration up
 * to that same banner, i.e. the entire raw-pointer half -- and holds every
 * scopy_* / copy_* in it to the same rule, wherever the header files it (they
 * are spread over five different banners today, two of them "BLAS").
 */
TEST(CudaCopyOpTable, CensusCoversCopyOps) {
#ifndef NNTR_COMPUTE_OPS_HEADER
  GTEST_SKIP() << "NNTR_COMPUTE_OPS_HEADER not defined by the build";
#else
  std::ifstream in(NNTR_COMPUTE_OPS_HEADER);
  if (!in) {
    GTEST_SKIP() << "compute_ops.h not readable at " << NNTR_COMPUTE_OPS_HEADER
                 << " (installed test run); the compile-time half still ran";
  }
  std::ostringstream buf;
  buf << in.rdbuf();
  const std::string src = buf.str();

  // Raw-pointer half: the class declaration through to the whole-op banner.
  // Carved on the RAW text, like the whole-op census, because the end anchor
  // is itself inside a comment.
  const size_t begin = src.find("class ComputeOps {");
  ASSERT_NE(begin, std::string::npos)
    << "cannot find `class ComputeOps {` in compute_ops.h -- re-anchor this"
       " census.";
  const size_t end = src.find("Whole-op (Tensor-level) ops", begin);
  ASSERT_NE(end, std::string::npos)
    << "the whole-op section banner moved in compute_ops.h; the copy census"
       " cannot delimit the raw-pointer half -- re-anchor it.";

  const std::vector<std::string> declared =
    declaredVirtuals(stripComments(src.substr(begin, end - begin)));
  ASSERT_FALSE(declared.empty())
    << "parsed zero virtuals out of the raw-pointer section -- the parser, not"
       " the table, is broken.";

  std::set<std::string> known;
  for (const auto &op : kCopyOps)
    known.insert(op.name);

  reconcile(declared, known, "kCopyOps", isCopyOp);
#endif
}

#endif // ENABLE_CUDA

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

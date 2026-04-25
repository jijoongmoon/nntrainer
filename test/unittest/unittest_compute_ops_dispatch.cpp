// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   unittest_compute_ops_dispatch.cpp
 * @date   25 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Verify that a per-Context ComputeOps table installed into
 *         ContextData actually reaches Tensor::dot / multiply / add
 *         through the Tensor's attached ContextData. This is the
 *         end-to-end check that vendor backend dispatch works without
 *         any preprocessor branches at the call site.
 */

#include <compute_ops.h>
#include <context_data.h>
#include <gtest/gtest.h>
#include <tensor.h>

#include <atomic>
#include <memory>

namespace {

/**
 * @brief Per-test counters incremented by mock ComputeOps wrappers.
 *
 * The mock wrappers forward to the global ops so result correctness is
 * preserved; the counters confirm dispatch reached the mock.
 */
struct CallCounters {
  std::atomic<int> sgemm{0};
  std::atomic<int> sgemv{0};
  std::atomic<int> ele_mul{0};
  std::atomic<int> ele_add{0};
  std::atomic<int> scopy{0};
};

CallCounters *g_counters = nullptr;

void mock_sgemm(unsigned int o, bool tA, bool tB, unsigned int M,
                unsigned int N, unsigned int K, float a, const float *A,
                unsigned int lda, const float *B, unsigned int ldb, float b,
                float *C, unsigned int ldc) {
  if (g_counters)
    g_counters->sgemm++;
  nntrainer::getComputeOps()->sgemm_fp32(o, tA, tB, M, N, K, a, A, lda, B, ldb,
                                         b, C, ldc);
}

void mock_sgemv(unsigned int o, bool tA, unsigned int M, unsigned int N,
                float a, const float *A, unsigned int lda, const float *X,
                unsigned int iX, float b, float *Y, unsigned int iY) {
  if (g_counters)
    g_counters->sgemv++;
  nntrainer::getComputeOps()->sgemv_fp32(o, tA, M, N, a, A, lda, X, iX, b, Y,
                                         iY);
}

void mock_ele_mul(unsigned int N, const float *X, const float *Y, float *Z,
                  float a, float b, unsigned int is, unsigned int os) {
  if (g_counters)
    g_counters->ele_mul++;
  nntrainer::getComputeOps()->ele_mul_fp32(N, X, Y, Z, a, b, is, os);
}

void mock_ele_add(unsigned int N, const float *X, const float *Y, float *Z,
                  float a, float b, unsigned int is, unsigned int os) {
  if (g_counters)
    g_counters->ele_add++;
  nntrainer::getComputeOps()->ele_add_fp32(N, X, Y, Z, a, b, is, os);
}

void mock_scopy(unsigned int N, const float *X, unsigned int iX, float *Y,
                unsigned int iY) {
  if (g_counters)
    g_counters->scopy++;
  nntrainer::getComputeOps()->scopy_fp32(N, X, iX, Y, iY);
}

/**
 * @brief Build a ComputeOps table that starts as a copy of the active
 *        backend, then overrides a few function pointers with the mocks.
 *
 * Copying the global table guarantees every other op still works (so
 * tensor allocate/copy paths used by the test setup don't crash).
 */
nntrainer::ComputeOps make_mock_ops() {
  nntrainer::ensureComputeOps();
  nntrainer::ComputeOps ops = *nntrainer::getComputeOps();
  ops.sgemm_fp32 = mock_sgemm;
  ops.sgemv_fp32 = mock_sgemv;
  ops.ele_mul_fp32 = mock_ele_mul;
  ops.ele_add_fp32 = mock_ele_add;
  ops.scopy_fp32 = mock_scopy;
  return ops;
}

class ComputeOpsDispatchTest : public ::testing::Test {
protected:
  void SetUp() override {
    counters = std::make_unique<CallCounters>();
    g_counters = counters.get();
    mock_ops = make_mock_ops();
    ct_data = std::make_shared<nntrainer::ContextData>();
    ct_data->setComputeOps(&mock_ops);
  }

  void TearDown() override { g_counters = nullptr; }

  std::unique_ptr<CallCounters> counters;
  nntrainer::ComputeOps mock_ops{};
  std::shared_ptr<nntrainer::ContextData> ct_data;
};

} // namespace

/**
 * @brief A Tensor with no attached ContextData should fall back to the
 *        global ops table; no mock counter increments.
 */
TEST_F(ComputeOpsDispatchTest, FallbackToGlobalWhenNoContextData) {
  nntrainer::Tensor a(1, 1, 4, 4); // 4x4
  nntrainer::Tensor b(1, 1, 4, 4);
  a.setValue(1.0f);
  b.setValue(1.0f);

  auto out = a.dot(b);

  EXPECT_EQ(counters->sgemm.load(), 0);
  EXPECT_EQ(counters->sgemv.load(), 0);
}

/**
 * @brief When a ContextData with a mock ComputeOps is attached to a
 *        Tensor, calling .dot dispatches through the mock.
 */
TEST_F(ComputeOpsDispatchTest, DotDispatchesThroughAttachedContextOps) {
  nntrainer::Tensor a(1, 1, 4, 4);
  nntrainer::Tensor b(1, 1, 4, 4);
  a.setValue(1.0f);
  b.setValue(1.0f);

  a.setContextData(ct_data);

  auto out = a.dot(b);

  // Either sgemm or sgemv must have been invoked (4x4 dot uses sgemm).
  EXPECT_GT(counters->sgemm.load() + counters->sgemv.load(), 0);
  // Result correctness preserved (mock forwards to global).
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
 *        same backend. Verifies the inheritance contract.
 */
TEST_F(ComputeOpsDispatchTest, ResultInheritsContextDataFromOperand) {
  nntrainer::Tensor a(1, 1, 4, 4);
  nntrainer::Tensor b(1, 1, 4, 4);
  a.setValue(1.0f);
  b.setValue(1.0f);

  a.setContextData(ct_data);

  auto first = a.dot(b);
  // first should now own ct_data via inheritContextDataTo.
  EXPECT_EQ(first.getContextData().get(), ct_data.get());

  int before = counters->sgemm.load() + counters->sgemv.load();
  auto second = first.dot(b);
  int after = counters->sgemm.load() + counters->sgemv.load();
  EXPECT_GT(after, before);
}

/**
 * @brief Replacing the attached ContextData with a different mock
 *        rebinds dispatch on subsequent calls. This is the "swap
 *        backend at runtime" property — important for hot-swapping
 *        between vendor contexts on the same Tensor.
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

  // Swap to a fresh ContextData carrying its OWN ComputeOps that does
  // NOT touch the original counters.
  nntrainer::ComputeOps fresh_ops = *nntrainer::getComputeOps();
  auto fresh_ct = std::make_shared<nntrainer::ContextData>();
  fresh_ct->setComputeOps(&fresh_ops);
  a.setContextData(fresh_ct);

  a.multiply(b, out2);
  // The original mock counter must not have advanced.
  EXPECT_EQ(counters->ele_mul.load(), after_first);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

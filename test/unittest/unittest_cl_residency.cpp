// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   unittest_cl_residency.cpp
 * @date   24 August 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Tests for the static residency plane: which tensors the memory
 *         planner places in device memory, and what a placed tensor hands a
 *         layer that has to bind it to a kernel.
 */

#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <basic_planner.h>
#include <cl_context.h>
#include <compute_ops.h>
#include <engine.h>
#include <mem_allocator.h>
#include <optimized_v1_planner.h>
#include <residency_policy.h>
#include <tensor_pool.h>

namespace {

constexpr auto FWD = nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN;
constexpr auto GPU = ml::train::LayerComputeEngine::GPU;
constexpr auto CPU = ml::train::LayerComputeEngine::CPU;

/** @brief the allocator the OpenCL context installs (shared virtual memory) */
std::shared_ptr<nntrainer::MemAllocator> clAllocator() {
  return nntrainer::Engine::Global()
    .getRegisteredContext("gpu")
    ->getMemAllocator();
}

nntrainer::TensorDim fp16Dim(unsigned int h, unsigned int w) {
  return nntrainer::TensorDim(
    1, 1, h, w, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16});
}

nntrainer::TensorDim fp32Dim(unsigned int h, unsigned int w) {
  return nntrainer::TensorDim(
    1, 1, h, w, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});
}

/**
 * @brief Build a one-source-one-view pool on the given allocator and allocate
 *        it, so the residency decision is the one the planner really took.
 */
nntrainer::Tensor *planOne(nntrainer::TensorPool &pool, const std::string &name,
                           const nntrainer::TensorDim &dim,
                           ml::train::LayerComputeEngine producer,
                           ml::train::LayerComputeEngine consumer) {
  auto *t = pool.request(name, dim, {0}, FWD, nntrainer::Initializer::NONE,
                         /*is_weight_grad=*/false, producer);
  pool.view(name + "_view", name, dim, {1}, FWD, 0, consumer);
  return t;
}

} // namespace

/**
 * @brief A tensor an OpenCL layer writes and only OpenCL layers read is placed
 *        in device memory, and hands out the buffer a kernel binds.
 */
TEST(ClResidency, gpu_written_gpu_read_tensor_is_device_resident) {
  nntrainer::TensorPool pool(false, "", "residency_gpu",
                             ml::train::ExecutionMode::INFERENCE,
                             clAllocator());
  auto *t = planOne(pool, "act", fp16Dim(4, 16), GPU, GPU);

  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  EXPECT_TRUE(t->isClMem());
  EXPECT_NE(t->getClMem(), nullptr);

  pool.deallocate();
}

/**
 * @brief One consumer that does not run on the device pulls the whole tensor
 *        back onto the shared plane: a placement has to be one every reader
 *        can reach, so it is never half applied.
 */
TEST(ClResidency, a_host_consumer_keeps_the_tensor_on_the_shared_plane) {
  nntrainer::TensorPool pool(false, "", "residency_mixed",
                             ml::train::ExecutionMode::INFERENCE,
                             clAllocator());
  auto *t = planOne(pool, "act", fp16Dim(4, 16), GPU, CPU);

  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  EXPECT_FALSE(t->isClMem());
  EXPECT_EQ(t->getClMem(), nullptr);
  EXPECT_NE(t->getData<_FP16>(), nullptr);

  pool.deallocate();
}

/**
 * @brief A host-written tensor stays where the host wrote it, whoever reads it.
 */
TEST(ClResidency, a_host_written_tensor_stays_on_the_shared_plane) {
  nntrainer::TensorPool pool(false, "", "residency_cpu_producer",
                             ml::train::ExecutionMode::INFERENCE,
                             clAllocator());
  auto *t = planOne(pool, "act", fp16Dim(4, 16), CPU, GPU);

  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  EXPECT_FALSE(t->isClMem());

  pool.deallocate();
}

/**
 * @brief The declared input boundary: a tensor a host producer uploads to the
 *        device plane itself may be placed there anyway, because the upload is
 *        the point at which the two agree.
 */
TEST(ClResidency, a_declared_input_boundary_is_raised_to_device_memory) {
  auto &policy = nntrainer::ResidencyPolicy::global();
  const std::string saved = policy.raise_patterns;
  policy.raise_patterns = "uploaded";

  nntrainer::TensorPool pool(false, "", "residency_raise",
                             ml::train::ExecutionMode::INFERENCE,
                             clAllocator());
  auto *raised = planOne(pool, "uploaded", fp16Dim(4, 16), CPU, GPU);
  auto *plain = planOne(pool, "other", fp16Dim(4, 16), CPU, GPU);

  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  EXPECT_TRUE(raised->isClMem());
  EXPECT_FALSE(plain->isClMem());

  pool.deallocate();
  policy.raise_patterns = saved;
}

/**
 * @brief A tensor the application excluded by name is kept off the device
 *        plane even when the heuristic would have placed it there.
 */
TEST(ClResidency, a_declared_exclusion_is_kept_on_the_shared_plane) {
  auto &policy = nntrainer::ResidencyPolicy::global();
  const std::string saved = policy.exclude_patterns;
  policy.exclude_patterns = "cache_";

  nntrainer::TensorPool pool(false, "", "residency_exclude",
                             ml::train::ExecutionMode::INFERENCE,
                             clAllocator());
  auto *excluded = planOne(pool, "cache_k", fp16Dim(4, 16), GPU, GPU);
  auto *plain = planOne(pool, "act", fp16Dim(4, 16), GPU, GPU);

  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  EXPECT_FALSE(excluded->isClMem());
  EXPECT_TRUE(plain->isClMem());

  pool.deallocate();
  policy.exclude_patterns = saved;
}

/**
 * @brief The kernels that read the device plane compute in FP16, so an FP32
 *        tensor keeps the plane both sides can address.
 */
TEST(ClResidency, an_fp32_tensor_stays_on_the_shared_plane) {
  nntrainer::TensorPool pool(false, "", "residency_fp32",
                             ml::train::ExecutionMode::INFERENCE,
                             clAllocator());
  auto *t = planOne(pool, "act", fp32Dim(4, 16), GPU, GPU);

  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  EXPECT_FALSE(t->isClMem());

  pool.deallocate();
}

/**
 * @brief On a host allocator there is one plane and nothing to place: the
 *        classification runs and every tensor keeps the pointer it had.
 */
TEST(ClResidency, a_host_pool_places_nothing_in_device_memory) {
  nntrainer::TensorPool pool;
  auto *t = planOne(pool, "act", fp16Dim(4, 16), GPU, GPU);

  pool.finalize(nntrainer::BasicPlanner(), 0, 2);
  pool.allocate();

  EXPECT_FALSE(t->isClMem());
  EXPECT_EQ(t->getClMem(), nullptr);
  EXPECT_NE(t->getData<_FP16>(), nullptr);

  pool.deallocate();
}

/**
 * @brief Tensors the planner placed at one offset have disjoint lifetimes and
 *        share one device buffer, so the reuse the planner intended is reuse
 *        of the same handle rather than two handles over one region.
 */
TEST(ClResidency, tensors_sharing_a_planner_offset_share_one_buffer) {
  nntrainer::TensorPool pool(false, "", "residency_reuse",
                             ml::train::ExecutionMode::INFERENCE,
                             clAllocator());
  auto *first = pool.request("first", fp16Dim(4, 16), {0}, FWD,
                             nntrainer::Initializer::NONE,
                             /*is_weight_grad=*/false, GPU);
  pool.view("first_view", "first", fp16Dim(4, 16), {1}, FWD, 0, GPU);
  auto *second = pool.request("second", fp16Dim(4, 16), {2}, FWD,
                              nntrainer::Initializer::NONE,
                              /*is_weight_grad=*/false, GPU);
  pool.view("second_view", "second", fp16Dim(4, 16), {3}, FWD, 0, GPU);

  /** OptimizedV1Planner reuses an offset across disjoint lifetimes; the two
   *  tensors above never overlap, so the planner places them together. */
  pool.finalize(nntrainer::OptimizedV1Planner(), 0, 4);
  pool.allocate();

  ASSERT_TRUE(first->isClMem());
  ASSERT_TRUE(second->isClMem());
  EXPECT_EQ(first->getClMem(), second->getClMem());

  pool.deallocate();
}

GTEST_API_ int main(int argc, char **argv) {
  nntrainer::init_backend();

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

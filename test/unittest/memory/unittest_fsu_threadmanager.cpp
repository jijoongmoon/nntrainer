// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file unittest_fsu_threadmanager.cpp
 * @date 21 March 2026
 * @brief FSU test using ThreadManager (replaces CacheLoader tests)
 * @see  https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <cache_pool.h>
#include <completion_token.h>
#include <optimized_v1_planner.h>
#include <thread_manager.h>

// ─── Test Fixture ───────────────────────────────────────────

class FSUThreadManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    pool = std::make_shared<nntrainer::CachePool>("tmp pool");
  }

  void TearDown() override { pool.reset(); }

  std::shared_ptr<nntrainer::CachePool> pool;
};

// ─── Direct CachePool + ThreadManager Tests ─────────────────

TEST_F(FSUThreadManagerTest, single_tensor_load_unload) {
  auto idx = pool->requestMemory(4, 1, 5, {1, 2, 3, 4, 5});
  pool->planLayout(nntrainer::OptimizedV1Planner());
  pool->allocate();
  auto mem = pool->getMemory(idx);
  EXPECT_EQ(mem->getAddr(), nullptr);

  auto &tm = nntrainer::ThreadManager::Global();

  // async load for order 1
  auto exec_ids = pool->getExecIDs(1);
  for (auto &id : exec_ids) {
    auto &elem = pool->getCacheElem(id);
    auto token = tm.submit([&, id] { pool->loadTensor(id); });
    elem.setLoadToken(std::move(token));
  }

  // wait for load
  for (auto &id : exec_ids)
    pool->getCacheElem(id).waitLoad();

  EXPECT_NE(mem->getAddr(), nullptr);

  // sync unload
  for (auto &id : exec_ids)
    pool->unloadTensor(id);

  EXPECT_EQ(mem->getAddr(), nullptr);
}

TEST_F(FSUThreadManagerTest, multi_tensor_load_unload) {
  auto idx1 = pool->requestMemory(4, 1, 5, {1, 2, 3, 4, 5});
  auto idx2 = pool->requestMemory(4, 3, 8, {3, 4, 5, 6, 7, 8});
  auto idx3 = pool->requestMemory(4, 2, 4, {2, 3, 4});

  pool->planLayout(nntrainer::OptimizedV1Planner());
  pool->allocate();
  EXPECT_EQ(pool->size(), 12u);

  auto mem1 = pool->getMemory(idx1);
  auto mem2 = pool->getMemory(idx2);
  auto mem3 = pool->getMemory(idx3);

  auto &tm = nntrainer::ThreadManager::Global();

  auto loadOrder = [&](unsigned int order) {
    auto ids = pool->getExecIDs(order);
    for (auto &id : ids) {
      auto &elem = pool->getCacheElem(id);
      auto token = tm.submit([&, id] { pool->loadTensor(id); });
      elem.setLoadToken(std::move(token));
    }
    for (auto &id : ids)
      pool->getCacheElem(id).waitLoad();
  };

  auto unloadOrder = [&](unsigned int order) {
    auto ids = pool->getExecIDs(order);
    for (auto &id : ids)
      pool->unloadTensor(id);
  };

  // order 1: only mem1 active
  loadOrder(1);
  EXPECT_NE(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);
  unloadOrder(1);

  // order 2: mem1 + mem3 active
  loadOrder(2);
  EXPECT_NE(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_NE(mem3->getAddr(), nullptr);
  unloadOrder(2);

  // order 3: all three active
  loadOrder(3);
  EXPECT_NE(mem1->getAddr(), nullptr);
  EXPECT_NE(mem2->getAddr(), nullptr);
  EXPECT_NE(mem3->getAddr(), nullptr);
  unloadOrder(3);

  // order 6: only mem2 active
  loadOrder(6);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_NE(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);
  unloadOrder(6);

  // order 9: none active
  loadOrder(9);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);
}

TEST_F(FSUThreadManagerTest, async_load_unload) {
  auto idx = pool->requestMemory(16, 1, 3, {1, 2, 3});
  pool->planLayout(nntrainer::OptimizedV1Planner());
  pool->allocate();
  auto mem = pool->getMemory(idx);

  auto &tm = nntrainer::ThreadManager::Global();

  // async load
  auto ids = pool->getExecIDs(1);
  for (auto &id : ids) {
    auto &elem = pool->getCacheElem(id);
    auto token = tm.submit([&, id] { pool->loadTensor(id); });
    elem.setLoadToken(std::move(token));
  }

  // async unload (must wait for load first)
  for (auto &id : ids) {
    auto &elem = pool->getCacheElem(id);
    elem.waitLoad();
    EXPECT_NE(mem->getAddr(), nullptr);
    auto token = tm.submit([&, id] { pool->unloadTensor(id); });
    elem.setUnloadToken(std::move(token));
  }

  // wait for unload
  for (auto &id : ids)
    pool->getCacheElem(id).waitUnload();

  EXPECT_EQ(mem->getAddr(), nullptr);
}

TEST_F(FSUThreadManagerTest, rapid_load_unload_cycles) {
  auto idx = pool->requestMemory(8, 1, 3, {1, 2, 3});
  pool->planLayout(nntrainer::OptimizedV1Planner());
  pool->allocate();
  auto mem = pool->getMemory(idx);

  auto &tm = nntrainer::ThreadManager::Global();

  for (int cycle = 0; cycle < 10; ++cycle) {
    // load
    auto ids = pool->getExecIDs(1);
    for (auto &id : ids) {
      auto &elem = pool->getCacheElem(id);
      auto token = tm.submit([&, id] { pool->loadTensor(id); });
      elem.setLoadToken(std::move(token));
    }
    for (auto &id : ids)
      pool->getCacheElem(id).waitLoad();

    EXPECT_NE(mem->getAddr(), nullptr) << "cycle " << cycle;

    // unload
    for (auto &id : ids)
      pool->unloadTensor(id);

    EXPECT_EQ(mem->getAddr(), nullptr) << "cycle " << cycle;
  }
}

TEST_F(FSUThreadManagerTest, completion_token_tracking) {
  auto idx = pool->requestMemory(4, 1, 2, {1, 2});
  pool->planLayout(nntrainer::OptimizedV1Planner());
  pool->allocate();

  auto &tm = nntrainer::ThreadManager::Global();
  auto ids = pool->getExecIDs(1);

  for (auto &id : ids) {
    auto &elem = pool->getCacheElem(id);

    // before load, token should be default (done)
    EXPECT_TRUE(elem.isLoadDone());

    auto token = tm.submit([&, id] { pool->loadTensor(id); });
    elem.setLoadToken(std::move(token));
  }

  // wait and verify
  for (auto &id : ids) {
    auto &elem = pool->getCacheElem(id);
    elem.waitLoad();
    EXPECT_TRUE(elem.isLoadDone());
  }

  // unload
  for (auto &id : ids)
    pool->unloadTensor(id);
}

TEST_F(FSUThreadManagerTest, concurrent_load_with_compute) {
  auto idx = pool->requestMemory(1024, 1, 3, {1, 2, 3});
  pool->planLayout(nntrainer::OptimizedV1Planner());
  pool->allocate();
  auto mem = pool->getMemory(idx);

  auto &tm = nntrainer::ThreadManager::Global();

  // start async load on I/O worker
  auto ids = pool->getExecIDs(1);
  for (auto &id : ids) {
    auto &elem = pool->getCacheElem(id);
    auto token = tm.submit([&, id] { pool->loadTensor(id); });
    elem.setLoadToken(std::move(token));
  }

  // while loading, do compute work on compute workers
  std::atomic<size_t> compute_sum{0};
  tm.parallel_for(0, 1000u, [&](size_t i) {
    compute_sum.fetch_add(i, std::memory_order_relaxed);
  });
  EXPECT_EQ(compute_sum.load(), 999 * 1000 / 2);

  // now wait for load
  for (auto &id : ids)
    pool->getCacheElem(id).waitLoad();

  EXPECT_NE(mem->getAddr(), nullptr);

  // unload
  for (auto &id : ids)
    pool->unloadTensor(id);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

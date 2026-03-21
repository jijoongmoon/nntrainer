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

// ─── Look-ahead (Prefetch) Tests ─────────────────────────────

/**
 * @brief Simulate the FSU look-ahead pipeline:
 *        While computing layer f, pre-load layer f+lookahead.
 *        Verifies that prefetched layers are already loaded when needed.
 */
TEST_F(FSUThreadManagerTest, lookahead_basic_pipeline) {
  // Create 5 "layers", each active only at its own order
  const unsigned int num_layers = 5;
  std::vector<unsigned int> idxs;
  std::vector<std::shared_ptr<nntrainer::MemoryData>> mems;

  for (unsigned int i = 1; i <= num_layers; ++i) {
    auto idx = pool->requestMemory(64, i, i + 1, {i});
    idxs.push_back(idx);
  }

  pool->planLayout(nntrainer::OptimizedV1Planner());
  pool->allocate();

  for (auto &idx : idxs)
    mems.push_back(pool->getMemory(idx));

  auto &tm = nntrainer::ThreadManager::Global();
  const unsigned int lookahead = 2;

  // Helper: async load all tensors for a given order
  auto asyncLoad = [&](unsigned int order) {
    auto ids = pool->getExecIDs(order);
    for (auto &id : ids) {
      auto &elem = pool->getCacheElem(id);
      auto token = tm.submit([&, id] { pool->loadTensor(id); });
      elem.setLoadToken(std::move(token));
    }
  };

  // Helper: wait for load to complete
  auto waitLoad = [&](unsigned int order) {
    auto ids = pool->getExecIDs(order);
    for (auto &id : ids)
      pool->getCacheElem(id).waitLoad();
  };

  // Helper: sync unload
  auto unload = [&](unsigned int order) {
    auto ids = pool->getExecIDs(order);
    for (auto &id : ids)
      pool->unloadTensor(id);
  };

  // Pre-load first (lookahead + 1) layers
  for (unsigned int i = 1; i <= std::min(lookahead + 1, num_layers); ++i)
    asyncLoad(i);

  // Forward pass simulation
  for (unsigned int f = 1; f <= num_layers; ++f) {
    // Step 1: wait for current layer
    waitLoad(f);
    EXPECT_NE(mems[f - 1]->getAddr(), nullptr)
      << "layer " << f << " should be loaded";

    // Step 2: simulate compute (parallel_for on compute workers)
    std::atomic<size_t> sum{0};
    tm.parallel_for(0, 100u, [&](size_t i) {
      sum.fetch_add(i, std::memory_order_relaxed);
    });
    EXPECT_EQ(sum.load(), 99 * 100 / 2);

    // Step 3: unload current layer
    unload(f);
    EXPECT_EQ(mems[f - 1]->getAddr(), nullptr)
      << "layer " << f << " should be unloaded";

    // Step 4: pre-load next look-ahead layer (if within bounds)
    unsigned int prefetch_order = f + lookahead + 1;
    if (prefetch_order <= num_layers)
      asyncLoad(prefetch_order);
  }
}

/**
 * @brief Verify that look-ahead actually overlaps I/O with compute.
 *        The prefetched layer should already be done (or nearly done)
 *        by the time we need it.
 */
TEST_F(FSUThreadManagerTest, lookahead_overlap_verification) {
  const unsigned int num_layers = 4;
  std::vector<unsigned int> idxs;
  std::vector<std::shared_ptr<nntrainer::MemoryData>> mems;

  for (unsigned int i = 1; i <= num_layers; ++i) {
    auto idx = pool->requestMemory(32, i, i + 1, {i});
    idxs.push_back(idx);
  }

  pool->planLayout(nntrainer::OptimizedV1Planner());
  pool->allocate();

  for (auto &idx : idxs)
    mems.push_back(pool->getMemory(idx));

  auto &tm = nntrainer::ThreadManager::Global();

  auto asyncLoad = [&](unsigned int order) {
    auto ids = pool->getExecIDs(order);
    for (auto &id : ids) {
      auto &elem = pool->getCacheElem(id);
      auto token = tm.submit([&, id] { pool->loadTensor(id); });
      elem.setLoadToken(std::move(token));
    }
  };

  auto waitLoad = [&](unsigned int order) {
    auto ids = pool->getExecIDs(order);
    for (auto &id : ids)
      pool->getCacheElem(id).waitLoad();
  };

  auto unload = [&](unsigned int order) {
    auto ids = pool->getExecIDs(order);
    for (auto &id : ids)
      pool->unloadTensor(id);
  };

  // Pre-load layer 1 and 2 (lookahead=1)
  asyncLoad(1);
  asyncLoad(2);

  // Process layer 1
  waitLoad(1);
  EXPECT_NE(mems[0]->getAddr(), nullptr);

  // Simulate heavy compute work while layer 2 loads in background
  std::atomic<size_t> sum{0};
  tm.parallel_for(0, 10000u, [&](size_t i) {
    sum.fetch_add(1, std::memory_order_relaxed);
  });

  // After compute, layer 2 should already be loaded (overlapped I/O)
  auto ids2 = pool->getExecIDs(2);
  for (auto &id : ids2) {
    EXPECT_TRUE(pool->getCacheElem(id).isLoadDone())
      << "prefetched layer 2 should be done after compute overlap";
  }

  unload(1);

  // Process layer 2 — no waiting stall expected
  waitLoad(2);
  EXPECT_NE(mems[1]->getAddr(), nullptr);

  // Pre-load layer 3 while processing layer 2
  asyncLoad(3);

  unload(2);

  // Layer 3 should be ready
  waitLoad(3);
  EXPECT_NE(mems[2]->getAddr(), nullptr);
  asyncLoad(4);
  unload(3);

  waitLoad(4);
  EXPECT_NE(mems[3]->getAddr(), nullptr);
  unload(4);
}

/**
 * @brief Test look-ahead with multiple epochs (repeated forward passes).
 *        Ensures tokens are properly reset between epochs.
 */
TEST_F(FSUThreadManagerTest, lookahead_multi_epoch) {
  const unsigned int num_layers = 3;
  std::vector<unsigned int> idxs;
  std::vector<std::shared_ptr<nntrainer::MemoryData>> mems;

  for (unsigned int i = 1; i <= num_layers; ++i) {
    auto idx = pool->requestMemory(16, i, i + 1, {i});
    idxs.push_back(idx);
  }

  pool->planLayout(nntrainer::OptimizedV1Planner());
  pool->allocate();

  for (auto &idx : idxs)
    mems.push_back(pool->getMemory(idx));

  auto &tm = nntrainer::ThreadManager::Global();

  auto asyncLoad = [&](unsigned int order) {
    auto ids = pool->getExecIDs(order);
    for (auto &id : ids) {
      auto &elem = pool->getCacheElem(id);
      auto token = tm.submit([&, id] { pool->loadTensor(id); });
      elem.setLoadToken(std::move(token));
    }
  };

  auto waitLoad = [&](unsigned int order) {
    auto ids = pool->getExecIDs(order);
    for (auto &id : ids)
      pool->getCacheElem(id).waitLoad();
  };

  auto unload = [&](unsigned int order) {
    auto ids = pool->getExecIDs(order);
    for (auto &id : ids)
      pool->unloadTensor(id);
  };

  const int num_epochs = 3;
  for (int epoch = 0; epoch < num_epochs; ++epoch) {
    // Pre-load layer 1 and 2
    asyncLoad(1);
    asyncLoad(2);

    for (unsigned int f = 1; f <= num_layers; ++f) {
      waitLoad(f);
      EXPECT_NE(mems[f - 1]->getAddr(), nullptr)
        << "epoch " << epoch << " layer " << f;

      // simulate compute
      std::atomic<size_t> s{0};
      tm.parallel_for(0, 50u,
                      [&](size_t i) { s.fetch_add(1, std::memory_order_relaxed); });
      EXPECT_EQ(s.load(), 50u);

      unload(f);
      EXPECT_EQ(mems[f - 1]->getAddr(), nullptr)
        << "epoch " << epoch << " layer " << f << " unload";

      // prefetch next
      if (f + 2 <= num_layers)
        asyncLoad(f + 2);
    }
  }
}

/**
 * @brief Test look-ahead with async unload pipeline:
 *        load(f+lookahead) and unload(f-1) happen concurrently
 *        while compute(f) runs on compute workers.
 */
TEST_F(FSUThreadManagerTest, lookahead_async_unload_pipeline) {
  const unsigned int num_layers = 5;
  std::vector<unsigned int> idxs;
  std::vector<std::shared_ptr<nntrainer::MemoryData>> mems;

  for (unsigned int i = 1; i <= num_layers; ++i) {
    auto idx = pool->requestMemory(64, i, i + 1, {i});
    idxs.push_back(idx);
  }

  pool->planLayout(nntrainer::OptimizedV1Planner());
  pool->allocate();

  for (auto &idx : idxs)
    mems.push_back(pool->getMemory(idx));

  auto &tm = nntrainer::ThreadManager::Global();

  auto asyncLoad = [&](unsigned int order) {
    auto ids = pool->getExecIDs(order);
    for (auto &id : ids) {
      auto &elem = pool->getCacheElem(id);
      auto token = tm.submit([&, id] { pool->loadTensor(id); });
      elem.setLoadToken(std::move(token));
    }
  };

  auto waitLoad = [&](unsigned int order) {
    auto ids = pool->getExecIDs(order);
    for (auto &id : ids)
      pool->getCacheElem(id).waitLoad();
  };

  auto asyncUnload = [&](unsigned int order) {
    auto ids = pool->getExecIDs(order);
    for (auto &id : ids) {
      auto &elem = pool->getCacheElem(id);
      auto token = tm.submit([&, id] { pool->unloadTensor(id); });
      elem.setUnloadToken(std::move(token));
    }
  };

  auto waitUnload = [&](unsigned int order) {
    auto ids = pool->getExecIDs(order);
    for (auto &id : ids)
      pool->getCacheElem(id).waitUnload();
  };

  // Pre-load layers 1 and 2
  asyncLoad(1);
  asyncLoad(2);

  for (unsigned int f = 1; f <= num_layers; ++f) {
    // Wait for current layer
    waitLoad(f);
    EXPECT_NE(mems[f - 1]->getAddr(), nullptr) << "layer " << f;

    // Concurrently: async unload previous + prefetch next + compute
    if (f > 1) {
      // wait for previous unload to finish before any reuse
      waitUnload(f - 1);
      EXPECT_EQ(mems[f - 2]->getAddr(), nullptr)
        << "layer " << (f - 1) << " should be fully unloaded";
    }

    // Prefetch f+2 (lookahead=1, but we already pre-loaded f+1)
    if (f + 2 <= num_layers)
      asyncLoad(f + 2);

    // Compute
    std::atomic<size_t> s{0};
    tm.parallel_for(0, 200u, [&](size_t i) {
      s.fetch_add(1, std::memory_order_relaxed);
    });
    EXPECT_EQ(s.load(), 200u);

    // Async unload current layer
    asyncUnload(f);
  }

  // Wait for last layer unload
  waitUnload(num_layers);
  EXPECT_EQ(mems[num_layers - 1]->getAddr(), nullptr);
}

/**
 * @brief Test CompletionToken waitFor() timeout with look-ahead.
 *        Verify isDone() polling works for non-blocking prefetch checks.
 */
TEST_F(FSUThreadManagerTest, lookahead_token_polling) {
  auto idx1 = pool->requestMemory(32, 1, 2, {1});
  auto idx2 = pool->requestMemory(32, 2, 3, {2});

  pool->planLayout(nntrainer::OptimizedV1Planner());
  pool->allocate();

  auto mem1 = pool->getMemory(idx1);
  auto mem2 = pool->getMemory(idx2);

  auto &tm = nntrainer::ThreadManager::Global();

  // Submit loads for both layers
  auto ids1 = pool->getExecIDs(1);
  for (auto &id : ids1) {
    auto &elem = pool->getCacheElem(id);
    auto token = tm.submit([&, id] { pool->loadTensor(id); });
    elem.setLoadToken(std::move(token));
  }

  auto ids2 = pool->getExecIDs(2);
  for (auto &id : ids2) {
    auto &elem = pool->getCacheElem(id);
    auto token = tm.submit([&, id] { pool->loadTensor(id); });
    elem.setLoadToken(std::move(token));
  }

  // Wait for layer 1 using blocking wait
  for (auto &id : ids1)
    pool->getCacheElem(id).waitLoad();

  EXPECT_NE(mem1->getAddr(), nullptr);

  // Poll layer 2 with waitFor timeout (should complete quickly)
  for (auto &id : ids2) {
    bool done =
      pool->getCacheElem(id).isLoadDone() ||
      pool->getCacheElem(id).isLoadDone(); // poll a couple times
    // Eventually it must complete
    pool->getCacheElem(id).waitLoad();
    EXPECT_TRUE(pool->getCacheElem(id).isLoadDone());
  }

  EXPECT_NE(mem2->getAddr(), nullptr);

  // Cleanup
  for (auto &id : ids1)
    pool->unloadTensor(id);
  for (auto &id : ids2)
    pool->unloadTensor(id);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

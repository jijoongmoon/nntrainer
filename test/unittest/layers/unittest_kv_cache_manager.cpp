// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   unittest_kv_cache_manager.cpp
 * @date   25 April 2026
 * @brief  Unit tests for KVCacheManager
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cstdio>
#include <gtest/gtest.h>
#include <string>

#include <kv_cache_manager.h>
#include <tensor.h>
#include <tensor_dim.h>

class KVCacheManagerTest : public ::testing::Test {
protected:
  static constexpr unsigned int NUM_LAYERS = 4;
  static constexpr unsigned int BATCH_SIZE = 2;
  static constexpr unsigned int MAX_SEQ_LEN = 128;
  static constexpr unsigned int NUM_HEADS_KV = 4;
  static constexpr unsigned int HEAD_DIM = 8;
  static constexpr unsigned int KV_WIDTH = NUM_HEADS_KV * HEAD_DIM;

  void SetUp() override {
    manager.allocate(NUM_LAYERS, BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS_KV,
                     HEAD_DIM, ml::train::TensorDim::DataType::FP32);
  }

  causallm::KVCacheManager manager;
};

TEST_F(KVCacheManagerTest, allocate_basic) {
  EXPECT_TRUE(manager.isAllocated());
  EXPECT_EQ(manager.getNumLayers(), NUM_LAYERS);
  EXPECT_EQ(manager.getMaxSeqLen(), MAX_SEQ_LEN);
  EXPECT_EQ(manager.getBatchSize(), BATCH_SIZE);
  EXPECT_EQ(manager.getKVWidth(), KV_WIDTH);
  EXPECT_EQ(manager.getPosition(), 0u);
}

TEST_F(KVCacheManagerTest, allocate_invalid_params) {
  causallm::KVCacheManager m;
  EXPECT_THROW(m.allocate(0, 1, 128, 4, 8), std::invalid_argument);
  EXPECT_THROW(m.allocate(4, 0, 128, 4, 8), std::invalid_argument);
  EXPECT_THROW(m.allocate(4, 1, 0, 4, 8), std::invalid_argument);
}

TEST_F(KVCacheManagerTest, cache_tensor_dimensions) {
  auto &k = manager.getKeyCache(0);
  auto &v = manager.getValueCache(0);

  EXPECT_EQ(k.batch(), BATCH_SIZE);
  EXPECT_EQ(k.channel(), 1u);
  EXPECT_EQ(k.height(), MAX_SEQ_LEN);
  EXPECT_EQ(k.width(), KV_WIDTH);

  EXPECT_EQ(v.batch(), BATCH_SIZE);
  EXPECT_EQ(v.channel(), 1u);
  EXPECT_EQ(v.height(), MAX_SEQ_LEN);
  EXPECT_EQ(v.width(), KV_WIDTH);
}

TEST_F(KVCacheManagerTest, position_management) {
  EXPECT_EQ(manager.getPosition(), 0u);

  manager.advance(10);
  EXPECT_EQ(manager.getPosition(), 10u);

  manager.advance(5);
  EXPECT_EQ(manager.getPosition(), 15u);

  manager.setPosition(50);
  EXPECT_EQ(manager.getPosition(), 50u);

  manager.reset();
  EXPECT_EQ(manager.getPosition(), 0u);
}

TEST_F(KVCacheManagerTest, position_bounds_check) {
  EXPECT_THROW(manager.setPosition(MAX_SEQ_LEN + 1), std::out_of_range);
  manager.setPosition(MAX_SEQ_LEN); // exactly at limit is ok

  manager.reset();
  manager.advance(MAX_SEQ_LEN);
  EXPECT_THROW(manager.advance(1), std::out_of_range);
}

TEST_F(KVCacheManagerTest, invalid_layer_idx) {
  EXPECT_THROW(manager.getKeyCache(NUM_LAYERS), std::out_of_range);
  EXPECT_THROW(manager.getValueCache(NUM_LAYERS), std::out_of_range);
  EXPECT_THROW(manager.getKeyCacheWriteView(NUM_LAYERS, 0, 1),
               std::out_of_range);
  EXPECT_THROW(manager.getValueCacheWriteView(NUM_LAYERS, 0, 1),
               std::out_of_range);
  EXPECT_THROW(manager.getKeyCacheReadView(NUM_LAYERS, 0, 1),
               std::out_of_range);
  EXPECT_THROW(manager.getValueCacheReadView(NUM_LAYERS, 0, 1),
               std::out_of_range);
}

TEST_F(KVCacheManagerTest, write_view_dimensions) {
  unsigned int step_size = 3;
  auto view = manager.getKeyCacheWriteView(0, 0, step_size);

  EXPECT_EQ(view.batch(), 1u);
  EXPECT_EQ(view.channel(), 1u);
  EXPECT_EQ(view.height(), step_size);
  EXPECT_EQ(view.width(), KV_WIDTH);
}

TEST_F(KVCacheManagerTest, read_view_dimensions) {
  unsigned int read_len = 10;
  auto view = manager.getKeyCacheReadView(0, 0, read_len);

  EXPECT_EQ(view.batch(), 1u);
  EXPECT_EQ(view.channel(), 1u);
  EXPECT_EQ(view.height(), read_len);
  EXPECT_EQ(view.width(), KV_WIDTH);
}

TEST_F(KVCacheManagerTest, write_view_points_to_correct_location) {
  // Write at position 0
  auto write_view = manager.getKeyCacheWriteView(0, 0, 1);
  float *write_ptr = write_view.getData<float>();

  // Read from position 0
  auto read_view = manager.getKeyCacheReadView(0, 0, 1);
  float *read_ptr = read_view.getData<float>();

  // Should point to same memory
  EXPECT_EQ(write_ptr, read_ptr);
}

TEST_F(KVCacheManagerTest, write_and_read_data_consistency) {
  // Write some data at position 0
  auto write_view = manager.getKeyCacheWriteView(0, 0, 1);
  float *data = write_view.getData<float>();
  for (unsigned int i = 0; i < KV_WIDTH; ++i) {
    data[i] = static_cast<float>(i + 1);
  }

  // Read it back
  auto read_view = manager.getKeyCacheReadView(0, 0, 1);
  float *read_data = read_view.getData<float>();
  for (unsigned int i = 0; i < KV_WIDTH; ++i) {
    EXPECT_FLOAT_EQ(read_data[i], static_cast<float>(i + 1));
  }
}

TEST_F(KVCacheManagerTest, sequential_write_positions) {
  // Simulate prefill: write 5 tokens
  auto &k_cache = manager.getKeyCache(0);
  float *cache_base = k_cache.getData<float>();

  auto view0 = manager.getKeyCacheWriteView(0, 0, 5);
  float *ptr0 = view0.getData<float>();
  EXPECT_EQ(ptr0, cache_base); // starts at beginning

  // Advance position
  manager.advance(5);

  // Write 1 more token
  auto view1 = manager.getKeyCacheWriteView(0, 0, 1);
  float *ptr1 = view1.getData<float>();
  EXPECT_EQ(ptr1, cache_base + 5 * KV_WIDTH); // offset by 5 tokens
}

TEST_F(KVCacheManagerTest, batch_offset_correct) {
  auto &k_cache = manager.getKeyCache(0);
  float *cache_base = k_cache.getData<float>();
  size_t feature_len = k_cache.getDim().getFeatureLen();

  // Batch 0
  auto view_b0 = manager.getKeyCacheWriteView(0, 0, 1);
  float *ptr_b0 = view_b0.getData<float>();
  EXPECT_EQ(ptr_b0, cache_base);

  // Batch 1
  auto view_b1 = manager.getKeyCacheWriteView(0, 1, 1);
  float *ptr_b1 = view_b1.getData<float>();
  EXPECT_EQ(ptr_b1, cache_base + feature_len);
}

TEST_F(KVCacheManagerTest, multi_layer_independence) {
  // Write different data to layer 0 and layer 1
  auto view_l0 = manager.getKeyCacheWriteView(0, 0, 1);
  auto view_l1 = manager.getKeyCacheWriteView(1, 0, 1);

  view_l0.getData<float>()[0] = 42.0f;
  view_l1.getData<float>()[0] = 99.0f;

  auto read_l0 = manager.getKeyCacheReadView(0, 0, 1);
  auto read_l1 = manager.getKeyCacheReadView(1, 0, 1);

  EXPECT_FLOAT_EQ(read_l0.getData<float>()[0], 42.0f);
  EXPECT_FLOAT_EQ(read_l1.getData<float>()[0], 99.0f);
}

TEST_F(KVCacheManagerTest, save_and_load) {
  // Write data to all layers
  for (unsigned int l = 0; l < NUM_LAYERS; ++l) {
    auto k_view = manager.getKeyCacheWriteView(l, 0, 3);
    auto v_view = manager.getValueCacheWriteView(l, 0, 3);
    float *kd = k_view.getData<float>();
    float *vd = v_view.getData<float>();
    for (unsigned int i = 0; i < 3 * KV_WIDTH; ++i) {
      kd[i] = static_cast<float>(l * 1000 + i);
      vd[i] = static_cast<float>(l * 1000 + i + 500);
    }
  }
  manager.advance(3);

  // Save
  std::string path = "/tmp/test_kv_cache.bin";
  manager.save(path);

  // Create a new manager and load
  causallm::KVCacheManager loaded;
  loaded.allocate(NUM_LAYERS, BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS_KV, HEAD_DIM,
                  ml::train::TensorDim::DataType::FP32);

  loaded.load(path, 3);
  EXPECT_EQ(loaded.getPosition(), 3u);

  // Verify data
  for (unsigned int l = 0; l < NUM_LAYERS; ++l) {
    auto k_read = loaded.getKeyCacheReadView(l, 0, 3);
    auto v_read = loaded.getValueCacheReadView(l, 0, 3);
    float *kd = k_read.getData<float>();
    float *vd = v_read.getData<float>();
    for (unsigned int i = 0; i < 3 * KV_WIDTH; ++i) {
      EXPECT_FLOAT_EQ(kd[i], static_cast<float>(l * 1000 + i))
        << "Key mismatch at layer=" << l << " i=" << i;
      EXPECT_FLOAT_EQ(vd[i], static_cast<float>(l * 1000 + i + 500))
        << "Value mismatch at layer=" << l << " i=" << i;
    }
  }

  // Cleanup
  std::remove(path.c_str());
}

TEST_F(KVCacheManagerTest, save_load_not_allocated) {
  causallm::KVCacheManager empty;
  EXPECT_THROW(empty.save("/tmp/test.bin"), std::runtime_error);
  EXPECT_THROW(empty.load("/tmp/test.bin", 1), std::runtime_error);
}

TEST_F(KVCacheManagerTest, write_view_overflow) {
  manager.setPosition(MAX_SEQ_LEN - 1);
  // Writing 1 should be ok
  EXPECT_NO_THROW(manager.getKeyCacheWriteView(0, 0, 1));
  // Writing 2 should overflow
  EXPECT_THROW(manager.getKeyCacheWriteView(0, 0, 2), std::out_of_range);
}

TEST_F(KVCacheManagerTest, typical_inference_flow) {
  // Simulate: prefill 10 tokens, then generate 5 tokens one by one

  // Prefill: write 10 tokens
  for (unsigned int l = 0; l < NUM_LAYERS; ++l) {
    for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
      auto k_write = manager.getKeyCacheWriteView(l, b, 10);
      auto v_write = manager.getValueCacheWriteView(l, b, 10);
      // Fill with identifiable data
      float *kd = k_write.getData<float>();
      for (unsigned int i = 0; i < 10 * KV_WIDTH; ++i) {
        kd[i] = static_cast<float>(l * 10000 + b * 1000 + i);
      }
    }
  }
  manager.advance(10);
  EXPECT_EQ(manager.getPosition(), 10u);

  // Generate: 5 tokens one by one
  for (unsigned int step = 0; step < 5; ++step) {
    unsigned int current_pos = manager.getPosition();
    for (unsigned int l = 0; l < NUM_LAYERS; ++l) {
      for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
        // Write new K/V
        auto k_write = manager.getKeyCacheWriteView(l, b, 1);
        float *kd = k_write.getData<float>();
        for (unsigned int i = 0; i < KV_WIDTH; ++i) {
          kd[i] = static_cast<float>(current_pos * 100 + l * 10 + i);
        }

        // Read all cached K for attention
        auto k_read = manager.getKeyCacheReadView(l, b, current_pos + 1);
        EXPECT_EQ(k_read.height(), current_pos + 1);
      }
    }
    manager.advance(1);
  }

  EXPECT_EQ(manager.getPosition(), 15u);

  // Verify first token of prefill is still intact (layer 0, batch 0)
  auto k_full = manager.getKeyCacheReadView(0, 0, 15);
  float *kd = k_full.getData<float>();
  EXPECT_FLOAT_EQ(kd[0], 0.0f); // l=0, b=0, i=0
  EXPECT_FLOAT_EQ(kd[1], 1.0f); // l=0, b=0, i=1
}

// Multi-session independence: two KVCacheManagers serve independent sessions
// in the same process — writes to one must not be visible from the other.
// This is the host-side property that makes mha_core's stateless / position-
// as-input design safe for concurrent / branching inference.
TEST_F(KVCacheManagerTest, multi_session_independence) {
  causallm::KVCacheManager session_b;
  session_b.allocate(NUM_LAYERS, BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS_KV,
                     HEAD_DIM, ml::train::TensorDim::DataType::FP32);

  // Session A writes 'A' marker at position 0
  for (unsigned int l = 0; l < NUM_LAYERS; ++l) {
    auto kw = manager.getKeyCacheWriteView(l, 0, 1);
    auto vw = manager.getValueCacheWriteView(l, 0, 1);
    for (unsigned int i = 0; i < KV_WIDTH; ++i) {
      kw.getData<float>()[i] = 100.0f + l;
      vw.getData<float>()[i] = 200.0f + l;
    }
  }
  manager.advance(1);

  // Session B writes 'B' marker at the same logical position 0
  for (unsigned int l = 0; l < NUM_LAYERS; ++l) {
    auto kw = session_b.getKeyCacheWriteView(l, 0, 1);
    auto vw = session_b.getValueCacheWriteView(l, 0, 1);
    for (unsigned int i = 0; i < KV_WIDTH; ++i) {
      kw.getData<float>()[i] = 500.0f + l;
      vw.getData<float>()[i] = 600.0f + l;
    }
  }
  session_b.advance(1);

  // Each session sees only its own data
  for (unsigned int l = 0; l < NUM_LAYERS; ++l) {
    auto a_k = manager.getKeyCacheReadView(l, 0, 1);
    auto a_v = manager.getValueCacheReadView(l, 0, 1);
    auto b_k = session_b.getKeyCacheReadView(l, 0, 1);
    auto b_v = session_b.getValueCacheReadView(l, 0, 1);
    for (unsigned int i = 0; i < KV_WIDTH; ++i) {
      EXPECT_FLOAT_EQ(a_k.getData<float>()[i], 100.0f + l);
      EXPECT_FLOAT_EQ(a_v.getData<float>()[i], 200.0f + l);
      EXPECT_FLOAT_EQ(b_k.getData<float>()[i], 500.0f + l);
      EXPECT_FLOAT_EQ(b_v.getData<float>()[i], 600.0f + l);
    }
  }

  EXPECT_EQ(manager.getPosition(), 1u);
  EXPECT_EQ(session_b.getPosition(), 1u);
}

// Multi-turn continuation: a single session's KV cache must accumulate across
// "turns" — turn 2 prefill must see turn 1's tokens at the start of cache and
// write its own at the position turn 1 left off at. This is the protocol
// CausalLM::bindPositionForCall + advanceKVCachePosition implements.
TEST_F(KVCacheManagerTest, multi_turn_continuation) {
  // Turn 1: write 4 tokens starting at position 0
  for (unsigned int t = 0; t < 4; ++t) {
    for (unsigned int l = 0; l < NUM_LAYERS; ++l) {
      auto kw = manager.getKeyCacheWriteView(l, 0, 1);
      for (unsigned int i = 0; i < KV_WIDTH; ++i) {
        kw.getData<float>()[i] = static_cast<float>(t * 10 + l);
      }
    }
    manager.advance(1);
  }
  EXPECT_EQ(manager.getPosition(), 4u);

  // Turn 2: write 3 tokens, must continue at position 4 (no reset)
  const unsigned int turn2_start = manager.getPosition();
  for (unsigned int t = 0; t < 3; ++t) {
    for (unsigned int l = 0; l < NUM_LAYERS; ++l) {
      auto kw = manager.getKeyCacheWriteView(l, 0, 1);
      for (unsigned int i = 0; i < KV_WIDTH; ++i) {
        kw.getData<float>()[i] = static_cast<float>(turn2_start + t + l * 100);
      }
    }
    manager.advance(1);
  }
  EXPECT_EQ(manager.getPosition(), 7u);

  // Read back: turn 1 marker survives at slots [0..4), turn 2 lives at [4..7)
  for (unsigned int l = 0; l < NUM_LAYERS; ++l) {
    auto k = manager.getKeyCacheReadView(l, 0, 7);
    const float *kd = k.getData<float>();
    for (unsigned int t = 0; t < 4; ++t) {
      EXPECT_FLOAT_EQ(kd[t * KV_WIDTH], static_cast<float>(t * 10 + l))
        << "turn1 token " << t << " layer " << l << " was clobbered";
    }
    for (unsigned int t = 0; t < 3; ++t) {
      EXPECT_FLOAT_EQ(kd[(4 + t) * KV_WIDTH],
                      static_cast<float>(4 + t + l * 100))
        << "turn2 token " << t << " layer " << l << " mis-written";
    }
  }
}

// Branching: two managers branch off the same shared prefix, then diverge.
// Demonstrates how a host can "fork" a generation safely with two KV caches.
TEST_F(KVCacheManagerTest, branching_from_shared_prefix) {
  const unsigned int prefix_len = 5;

  // Build a shared prefix in 'manager'
  for (unsigned int t = 0; t < prefix_len; ++t) {
    for (unsigned int l = 0; l < NUM_LAYERS; ++l) {
      auto kw = manager.getKeyCacheWriteView(l, 0, 1);
      for (unsigned int i = 0; i < KV_WIDTH; ++i)
        kw.getData<float>()[i] = static_cast<float>(t + l * 1000);
    }
    manager.advance(1);
  }
  manager.save("/tmp/test_kv_branch.bin", prefix_len);

  // Branch B: load prefix, then continue with branch-B specific tokens
  causallm::KVCacheManager branch_b;
  branch_b.allocate(NUM_LAYERS, BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS_KV, HEAD_DIM,
                    ml::train::TensorDim::DataType::FP32);
  branch_b.load("/tmp/test_kv_branch.bin", prefix_len);
  EXPECT_EQ(branch_b.getPosition(), prefix_len);

  // Continue 'manager' (branch A) with A-specific tokens
  for (unsigned int t = 0; t < 2; ++t) {
    for (unsigned int l = 0; l < NUM_LAYERS; ++l) {
      auto kw = manager.getKeyCacheWriteView(l, 0, 1);
      for (unsigned int i = 0; i < KV_WIDTH; ++i)
        kw.getData<float>()[i] = -1000.0f - t; // branch A marker
    }
    manager.advance(1);
  }

  // Continue branch B with B-specific tokens
  for (unsigned int t = 0; t < 2; ++t) {
    for (unsigned int l = 0; l < NUM_LAYERS; ++l) {
      auto kw = branch_b.getKeyCacheWriteView(l, 0, 1);
      for (unsigned int i = 0; i < KV_WIDTH; ++i)
        kw.getData<float>()[i] = +9000.0f + t; // branch B marker
    }
    branch_b.advance(1);
  }

  // Prefix is identical in both
  for (unsigned int l = 0; l < NUM_LAYERS; ++l) {
    auto a = manager.getKeyCacheReadView(l, 0, prefix_len);
    auto b = branch_b.getKeyCacheReadView(l, 0, prefix_len);
    for (unsigned int t = 0; t < prefix_len; ++t) {
      const float expected = static_cast<float>(t + l * 1000);
      EXPECT_FLOAT_EQ(a.getData<float>()[t * KV_WIDTH], expected);
      EXPECT_FLOAT_EQ(b.getData<float>()[t * KV_WIDTH], expected);
    }
  }
  // ...but the suffixes diverge
  for (unsigned int l = 0; l < NUM_LAYERS; ++l) {
    auto a = manager.getKeyCacheReadView(l, 0, prefix_len + 2);
    auto b = branch_b.getKeyCacheReadView(l, 0, prefix_len + 2);
    EXPECT_FLOAT_EQ(a.getData<float>()[prefix_len * KV_WIDTH], -1000.0f);
    EXPECT_FLOAT_EQ(a.getData<float>()[(prefix_len + 1) * KV_WIDTH], -1001.0f);
    EXPECT_FLOAT_EQ(b.getData<float>()[prefix_len * KV_WIDTH], 9000.0f);
    EXPECT_FLOAT_EQ(b.getData<float>()[(prefix_len + 1) * KV_WIDTH], 9001.0f);
  }

  std::remove("/tmp/test_kv_branch.bin");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

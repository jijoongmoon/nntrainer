// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gemma4_e2b_qnn.cpp
 * @brief  QNN model implementation for Gemma 4 E2B with PLE.
 *         Follows Gauss 3.6 KV-cache pattern; PLE handling layered on top.
 */

#include "gemma4_e2b_qnn.h"
#include "android_memory_allocator.h"
#include "generate_qnn_utils.h"

#include <llm_util.hpp>
#include "api/streamer.h"
#include <model.h>

#include <app_context.h>
#include <engine.h>
#include <factory.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <unordered_map>
#include <utility>

#include "json.hpp"

using namespace causallm;

// =====================================================================
// Anonymous namespace helpers
// =====================================================================
namespace {

bool starts_with(const std::string &v, const std::string &p) {
  return v.compare(0, p.size(), p) == 0;
}

bool is_absolute_path(const std::string &path) {
  return !path.empty() && path[0] == '/';
}

std::string dirname(const std::string &path) {
  auto pos = path.find_last_of('/');
  return (pos == std::string::npos) ? std::string() : path.substr(0, pos);
}

std::string rebase_relative_to_model_file(const std::string &path,
                                          const std::string &model_file) {
  if (path.empty() || is_absolute_path(path)) return path;
  auto base = dirname(model_file);
  if (base.empty()) return path;
  return base + "/" + path;
}

int find_tensor_index_or_minus_one(const TensorInfoList &tensor_infos,
                                   const std::string &name) {
  for (size_t i = 0; i < tensor_infos.size(); ++i) {
    if (tensor_infos[i].first == name) return (int)i;
  }
  return -1;
}

std::string kv_output_to_input_name(const std::string &out_name) {
  if (out_name.size() >= 4 &&
      out_name.compare(out_name.size() - 4, 4, "_out") == 0) {
    return out_name.substr(0, out_name.size() - 4) + "_in";
  }
  return out_name;
}

// Window copy used by sync_generation_kv_cache_to_prefill().
void copy_kv_cache_window(uint8_t *dest, int dest_row_length,
                          const uint8_t *src, int src_row_length,
                          int history_length, bool is_key, int num_columns) {
  if (!dest || !src || history_length <= 0 ||
      dest_row_length <= 0 || src_row_length <= 0) return;
  const int available  = std::min(history_length, src_row_length);
  const int copy_len   = std::min(available, dest_row_length);
  const int src_start  = available - copy_len;
  const bool tail      = (history_length >= src_row_length) &&
                         (dest_row_length > copy_len);
  const int dest_start = tail ? dest_row_length - copy_len : 0;

  if (is_key) {
    for (int col = 0; col < num_columns; ++col) {
      std::memcpy(dest + col * dest_row_length + dest_start,
                  src + col * src_row_length + src_start, copy_len);
    }
  } else {
    std::memcpy(dest + dest_start * num_columns,
                src + src_start * num_columns,
                copy_len * num_columns);
  }
}

// PLE 4-bit packed → uint16 (QNN consumer space) two-step requant.
// ufixed8 path: f = (q4bit + lut_offset) * lut_scale.
inline void dequant_nibbles_requant_u16(const uint8_t *packed, size_t elems,
                                        float lut_scale, int lut_offset,
                                        float out_scale, int out_offset,
                                        uint16_t *dst) {
  const float inv_out = 1.0f / out_scale;
  auto requant = [&](uint8_t nib) -> uint16_t {
    const float f = (static_cast<float>(nib) + lut_offset) * lut_scale;
    int q = static_cast<int>(std::lrintf(f * inv_out)) - out_offset;
    return static_cast<uint16_t>(std::max(0, std::min(65535, q)));
  };
  const size_t whole = elems / 2;
  for (size_t i = 0; i < whole; ++i) {
    const uint8_t b = packed[i];
    dst[2 * i]     = requant(b & 0x0F);
    dst[2 * i + 1] = requant((b >> 4) & 0x0F);
  }
  if (elems & 1) dst[2 * whole] = requant(packed[whole] & 0x0F);
}

// Sign-extend a 4-bit value (0..15 → -8..7).
inline int s4(unsigned nib) {
  return (nib & 0x8u) ? static_cast<int>(nib) - 16 : static_cast<int>(nib);
}

// PLE sfixed4 (per-row-per-layer) → uint16 requant. f = s4(nib) * row_scale.
inline void dequant_sfixed4_requant_u16(const uint8_t *packed, size_t elems,
                                         float row_scale,
                                         float out_scale, int out_offset,
                                         uint16_t *dst) {
  const float inv_out = 1.0f / out_scale;
  auto requant = [&](unsigned nib) -> uint16_t {
    const float f = static_cast<float>(s4(nib)) * row_scale;
    int q = static_cast<int>(std::lrintf(f * inv_out)) - out_offset;
    return static_cast<uint16_t>(std::max(0, std::min(65535, q)));
  };
  const size_t whole = elems / 2;
  for (size_t i = 0; i < whole; ++i) {
    const uint8_t b = packed[i];
    dst[2 * i]     = requant(b & 0x0F);
    dst[2 * i + 1] = requant((b >> 4) & 0x0F);
  }
  if (elems & 1) dst[2 * whole] = requant(packed[whole] & 0x0F);
}

} // namespace

// =====================================================================
// Auto-registration
// =====================================================================
__attribute__((constructor)) static void register_custom_models() {
  causallm::Factory::Instance().registerModel(
      "Gemma4_E2B_QNN",
      [](causallm::json cfg, causallm::json generation_cfg,
         causallm::json nntr_cfg) {
        return std::make_unique<causallm::Gemma4_E2B_QNN>(
            cfg, generation_cfg, nntr_cfg);
      });
}

// =====================================================================
// PLE methods (dual-mode: 4-bit manifest OR raw uint16 bin)
// =====================================================================
namespace {
inline bool ends_with(const std::string &s, const std::string &suf) {
  return s.size() >= suf.size() &&
         0 == s.compare(s.size() - suf.size(), suf.size(), suf);
}
} // namespace

void Gemma4_E2B_QNN::open_ple_file_() {
  if (ple_file_name.empty()) return;

  ple_is_4bit_ = ends_with(ple_file_name, ".json");
  ple_per_layer_ = 256;

  if (ple_is_4bit_) {
    // ── 4-bit manifest: dispatch by `datatype` (ufixed8 / sfixed4) ──
    std::ifstream mf(ple_file_name);
    if (!mf.is_open())
      throw std::runtime_error("Failed to open PLE manifest: " + ple_file_name);
    json j; mf >> j;

    const std::string lut_rel  = j.at("lut-path").get<std::string>();
    const int row_elems        = j.at("size").get<int>();
    const std::string datatype = j.value("datatype", std::string("ufixed8"));
    const auto &qp             = j.at("quant-param");

    ple_is_signed4_ = (datatype == "sfixed4");
    if (!ple_is_signed4_ && datatype != "ufixed8")
      throw std::runtime_error("PLE: unsupported datatype: " + datatype);

    ple_row_elems_ = static_cast<size_t>(row_elems);
    ple_row_bytes_ = (ple_row_elems_ + 1) / 2;
    ple_layers_    = ple_row_elems_ / ple_per_layer_;

    if (ple_layers_ * ple_per_layer_ != ple_row_elems_)
      throw std::runtime_error("PLE 'size' not divisible by 256");
    if (generation_per_layer_dst_.size() > ple_layers_)
      throw std::runtime_error("PLE layer count too small");

    if (ple_is_signed4_) {
      // Per-row-per-layer scale array; shape [vocab][layers] flat in
      // row-major. No offset (symmetric).
      const auto &scale_arr = qp.at("scale");
      if (!scale_arr.is_array())
        throw std::runtime_error(
          "PLE sfixed4: quant-param.scale must be an array");
      ple_row_layer_scales_.clear();
      ple_row_layer_scales_.reserve(scale_arr.size());
      for (const auto &v : scale_arr)
        ple_row_layer_scales_.push_back(v.get<float>());
      // Validate shape: must be a multiple of ple_layers_; vocab inferred.
      if (ple_row_layer_scales_.size() % ple_layers_ != 0)
        throw std::runtime_error(
          "PLE sfixed4: scale array length not divisible by num_layers");
      ple_scale_  = 1.0f; // unused
      ple_offset_ = 0;    // unused
    } else {
      ple_scale_  = qp.at("scale").get<float>();
      ple_offset_ = qp.at("offset").get<int>();
    }

    std::string lut_abs =
        rebase_relative_to_model_file(lut_rel, ple_file_name);

    ple_fd_ = open(lut_abs.c_str(), O_RDONLY);
    if (ple_fd_ < 0)
      throw std::runtime_error("open PLE bin: " + lut_abs);
    struct stat st;
    if (fstat(ple_fd_, &st) < 0) {
      ::close(ple_fd_); ple_fd_ = -1;
      throw std::runtime_error("stat PLE bin: " + lut_abs);
    }
    ple_file_size_ = static_cast<size_t>(st.st_size);
    if (ple_file_size_ % ple_row_bytes_ != 0) {
      ::close(ple_fd_); ple_fd_ = -1;
      throw std::runtime_error("PLE bin size not multiple of row bytes");
    }

    // For sfixed4 the scale array's vocab dim must match the bin's row
    // count; for ufixed8 there is no per-row scale to validate.
    if (ple_is_signed4_) {
      const size_t expected_vocab = ple_file_size_ / ple_row_bytes_;
      const size_t scale_vocab    = ple_row_layer_scales_.size() / ple_layers_;
      if (scale_vocab != expected_vocab)
        throw std::runtime_error(
          "PLE sfixed4 scale vocab=" + std::to_string(scale_vocab) +
          " != bin vocab=" + std::to_string(expected_vocab));
    }

    void *m = mmap(nullptr, ple_file_size_, PROT_READ, MAP_PRIVATE, ple_fd_, 0);
    if (m == MAP_FAILED) {
      ::close(ple_fd_); ple_fd_ = -1;
      throw std::runtime_error("mmap PLE bin: " + lut_abs);
    }
    ple_mmap_ = static_cast<const uint8_t *>(m);
#ifdef POSIX_MADV_RANDOM
    posix_madvise((void *)ple_mmap_, ple_file_size_, POSIX_MADV_RANDOM);
#endif
    if (ple_is_signed4_) {
      std::cout << "[PLE] sfixed4 (rowwise+layerwise) mmaped " << lut_abs
                << " rows=" << (ple_file_size_ / ple_row_bytes_)
                << " layers=" << ple_layers_ << " per_layer=" << ple_per_layer_
                << " scales=" << ple_row_layer_scales_.size() << std::endl;
    } else {
      std::cout << "[PLE] ufixed8 (tensorwise) mmaped " << lut_abs
                << " rows=" << (ple_file_size_ / ple_row_bytes_)
                << " layers=" << ple_layers_ << " per_layer=" << ple_per_layer_
                << " scale=" << ple_scale_ << " offset=" << ple_offset_
                << std::endl;
    }
    return;
  }

  // ── raw UINT16: row = ple_layers * 256 uint16, no manifest ──
  // Derive layer count from the generation graph's per_layer_inputs_*
  // count (collected before open_ple_file_() is called).
  ple_layers_    = generation_per_layer_dst_.size();
  if (ple_layers_ == 0)
    throw std::runtime_error(
      "PLE raw uint16: no per_layer slots collected from generation graph");
  ple_row_elems_ = ple_layers_ * ple_per_layer_;
  ple_row_bytes_ = ple_row_elems_ * sizeof(uint16_t);

  ple_fd_ = open(ple_file_name.c_str(), O_RDONLY);
  if (ple_fd_ < 0)
    throw std::runtime_error("open PLE bin: " + ple_file_name);
  struct stat st;
  if (fstat(ple_fd_, &st) < 0) {
    ::close(ple_fd_); ple_fd_ = -1;
    throw std::runtime_error("stat PLE bin: " + ple_file_name);
  }
  ple_file_size_ = static_cast<size_t>(st.st_size);
  if (ple_file_size_ % ple_row_bytes_ != 0) {
    ::close(ple_fd_); ple_fd_ = -1;
    throw std::runtime_error(
      "PLE raw uint16: file size not multiple of row bytes (expected "
      + std::to_string(ple_row_bytes_) + ")");
  }

  void *m = mmap(nullptr, ple_file_size_, PROT_READ, MAP_PRIVATE, ple_fd_, 0);
  if (m == MAP_FAILED) {
    ::close(ple_fd_); ple_fd_ = -1;
    throw std::runtime_error("mmap PLE bin: " + ple_file_name);
  }
  ple_u16_mmap_ = static_cast<const uint16_t *>(m);
  ple_mmap_     = static_cast<const uint8_t *>(m); // alias for cleanup
#ifdef POSIX_MADV_RANDOM
  posix_madvise(m, ple_file_size_, POSIX_MADV_RANDOM);
#endif
  std::cout << "[PLE] raw u16 mmaped " << ple_file_name
            << " rows=" << (ple_file_size_ / ple_row_bytes_)
            << " layers=" << ple_layers_ << " per_layer=" << ple_per_layer_
            << " (no requant)" << std::endl;
}

void Gemma4_E2B_QNN::close_ple_file_() {
  if (ple_mmap_) {
    munmap((void *)ple_mmap_, ple_file_size_);
    ple_mmap_     = nullptr;
    ple_u16_mmap_ = nullptr;
  }
  if (ple_fd_ >= 0) { ::close(ple_fd_); ple_fd_ = -1; }
}

void Gemma4_E2B_QNN::fill_prefill_ple_chunk_(const std::vector<int> &tokens,
                                             int chunk_idx, int chunk_len) {
  if (!ple_mmap_) return;
  const size_t L_pre = prefill_per_layer_dst_.size();
  const size_t per_layer_elems = ple_per_layer_;
  const int    chunk_size_tokens = context_size;

  // The PLE binary is laid out per model-layer (`ple_layers_` chunks of
  // `per_layer_elems` per row). The prefill graph may expose a SUBSET of
  // model layers via `per_layer_inputs_N`, so source rows MUST be indexed
  // by the parsed N (model layer index), not by the dense slot index `l`.
  const int *pre_layer_idx = prefill_per_layer_model_index_.data();

  if (ple_is_4bit_) {
    const size_t per_layer_bytes = per_layer_elems / 2;
    if (ple_is_signed4_) {
      // sfixed4: per-row-per-layer scale lookup, signed nibble decode.
      const float *scales = ple_row_layer_scales_.data();
      for (int t = 0; t < chunk_size_tokens; ++t) {
        const int abs_idx  = chunk_idx * chunk_size_tokens + t;
        const int token_id = (t < chunk_len) ? tokens[abs_idx] : padding_token;
        const uint8_t *row = ple_mmap_ + (size_t)token_id * ple_row_bytes_;
        const float *row_scales =
            scales + (size_t)token_id * ple_layers_;
        for (size_t l = 0; l < L_pre; ++l) {
          const size_t ml = (size_t)pre_layer_idx[l];
          uint16_t *dst =
              prefill_per_layer_dst_[l] + (size_t)t * per_layer_elems;
          dequant_sfixed4_requant_u16(
              row + ml * per_layer_bytes, per_layer_elems, row_scales[ml],
              prefill_per_layer_scale_[l],
              prefill_per_layer_offset_[l], dst);
        }
      }
    } else {
      for (int t = 0; t < chunk_size_tokens; ++t) {
        const int abs_idx  = chunk_idx * chunk_size_tokens + t;
        const int token_id = (t < chunk_len) ? tokens[abs_idx] : padding_token;
        const uint8_t *row = ple_mmap_ + (size_t)token_id * ple_row_bytes_;
        for (size_t l = 0; l < L_pre; ++l) {
          const size_t ml = (size_t)pre_layer_idx[l];
          uint16_t *dst =
              prefill_per_layer_dst_[l] + (size_t)t * per_layer_elems;
          dequant_nibbles_requant_u16(row + ml * per_layer_bytes, per_layer_elems,
                                      ple_scale_, ple_offset_,
                                      prefill_per_layer_scale_[l],
                                      prefill_per_layer_offset_[l], dst);
        }
      }
    }
  } else {
    // raw uint16: per-layer slice memcpy. Source already in consumer space.
    for (int t = 0; t < chunk_size_tokens; ++t) {
      const int abs_idx  = chunk_idx * chunk_size_tokens + t;
      const int token_id = (t < chunk_len) ? tokens[abs_idx] : padding_token;
      const uint16_t *row = ple_u16_mmap_ + (size_t)token_id * ple_row_elems_;
      for (size_t l = 0; l < L_pre; ++l) {
        const size_t ml = (size_t)pre_layer_idx[l];
        uint16_t *dst = prefill_per_layer_dst_[l] + (size_t)t * per_layer_elems;
        std::memcpy(dst, row + ml * per_layer_elems,
                    per_layer_elems * sizeof(uint16_t));
      }
    }
  }
}

void Gemma4_E2B_QNN::fill_generation_ple_(int token_id) {
  if (!ple_mmap_) return;
  const size_t L_gen = generation_per_layer_dst_.size();
  const size_t per_layer_elems = ple_per_layer_;
  const int *gen_layer_idx = generation_per_layer_model_index_.data();

  if (ple_is_4bit_) {
    const size_t per_layer_bytes = per_layer_elems / 2;
    const uint8_t *row = ple_mmap_ + (size_t)token_id * ple_row_bytes_;
    if (ple_is_signed4_) {
      const float *row_scales =
          ple_row_layer_scales_.data() + (size_t)token_id * ple_layers_;
      for (size_t l = 0; l < L_gen; ++l) {
        const size_t ml = (size_t)gen_layer_idx[l];
        dequant_sfixed4_requant_u16(
            row + ml * per_layer_bytes, per_layer_elems, row_scales[ml],
            generation_per_layer_scale_[l],
            generation_per_layer_offset_[l],
            generation_per_layer_dst_[l]);
      }
      // ── Debug: dump token's first decoded nibbles + scales (once) ──
      static bool dbg_done = false;
      if (!dbg_done) {
        dbg_done = true;
        std::cout << "[PLE-S4-DBG] token=" << token_id
                  << " row_offset=" << (size_t)token_id * ple_row_bytes_
                  << "\n[PLE-S4-DBG] L0 row_scale=" << row_scales[0]
                  << " L1=" << row_scales[1]
                  << " L17=" << row_scales[17]
                  << " L34=" << row_scales[34] << "\n";
        std::cout << "[PLE-S4-DBG] L0 raw bytes [0..7]: ";
        for (int i = 0; i < 8; ++i)
          std::cout << std::hex << (int)row[i] << " ";
        std::cout << std::dec
                  << "\n[PLE-S4-DBG] L0 nibbles s4 [0..15]: ";
        for (int i = 0; i < 8; ++i) {
          int lo = s4(row[i] & 0x0F);
          int hi = s4((row[i] >> 4) & 0x0F);
          std::cout << lo << " " << hi << " ";
        }
        std::cout << "\n[PLE-S4-DBG] L0 dst u16 [0..7]: ";
        for (int i = 0; i < 8; ++i)
          std::cout << generation_per_layer_dst_[0][i] << " ";
        std::cout << "\n[PLE-S4-DBG] L0 consumer scale="
                  << generation_per_layer_scale_[0]
                  << " offset=" << generation_per_layer_offset_[0]
                  << "\n";
      }
    } else {
      for (size_t l = 0; l < L_gen; ++l) {
        const size_t ml = (size_t)gen_layer_idx[l];
        dequant_nibbles_requant_u16(row + ml * per_layer_bytes, per_layer_elems,
                                    ple_scale_, ple_offset_,
                                    generation_per_layer_scale_[l],
                                    generation_per_layer_offset_[l],
                                    generation_per_layer_dst_[l]);
      }
    }
  } else {
    const uint16_t *row = ple_u16_mmap_ + (size_t)token_id * ple_row_elems_;
    for (size_t l = 0; l < L_gen; ++l) {
      const size_t ml = (size_t)gen_layer_idx[l];
      std::memcpy(generation_per_layer_dst_[l],
                  row + ml * per_layer_elems,
                  per_layer_elems * sizeof(uint16_t));
    }
  }
}

// =====================================================================
// Destructor
// =====================================================================
Gemma4_E2B_QNN::~Gemma4_E2B_QNN() {
  close_ple_file_();
  this->prefill_kv_zero_byte_.clear();
}

// =====================================================================
// KV cache helpers
// =====================================================================
void Gemma4_E2B_QNN::initialize_kv_cache() {
  kv_len = 0;
  conversation_started_ = false;
  for (int i = 0; i < (int)kvs.size(); ++i) {
    std::memcpy(kvs[i], fresh_kvs[i], kv_sizes[i]);
  }
  reset_prefill_kv_cache_inputs();
}

void Gemma4_E2B_QNN::reset_prefill_kv_cache_inputs() {
  for (int i = 0; i < (int)prefill_kvs.size(); ++i) {
    std::fill_n(prefill_kvs[i], prefill_kv_sizes[i],
		prefill_kv_zero_byte_[i]);
  }
}

void Gemma4_E2B_QNN::sync_generation_kv_cache_to_prefill() {
  reset_prefill_kv_cache_inputs();
  if (kv_len <= 0) return;

#pragma omp parallel for
  for (int i = 0; i < (int)prefill_kvs.size(); ++i) {
    int gen_idx = prefill_to_generation_kv_indices[i];
    int gen_layer = gen_idx / 2; // Gemma 4: 2 KV per layer
    if (gen_idx < 0 || gen_idx >= (int)kvs.size() ||
        gen_layer < 0 || gen_layer >= (int)kv_row_lengths.size())
      continue;

    copy_kv_cache_window(prefill_kvs[i], prefill_kv_row_lengths[i],
                         (uint8_t *)kvs[gen_idx],
                         kv_row_lengths[gen_layer], kv_len,
                         prefill_kv_is_key[i] != 0, kv_columns[gen_layer]);
  }
}

// =====================================================================
// initialize()
// =====================================================================
void Gemma4_E2B_QNN::initialize() {
  Quick_Dot_AI_QNN::initialize();
  LOGD("Quick_Dot_AI_QNN::initialize() done");

  std::string prefill_graph    = graphs_to_use[0];
  std::string generation_graph = graphs_to_use[1];

  auto &prefill_graph_info    = models[prefill_graph].graph_info;
  auto &generation_graph_info = models[generation_graph].graph_info;
  auto &prefill_inputs        = models[prefill_graph].model_inputs;
  auto &generation_inputs     = models[generation_graph].model_inputs;

  if (prefill_inputs.size() != prefill_graph_info.raw_inputs.size())
    throw std::runtime_error("prefill input count mismatch");
  if (generation_inputs.size() != generation_graph_info.raw_inputs.size())
    throw std::runtime_error("generation input count mismatch");

  // ── Mask / RoPE element counts ──
  prefill_attention_mask_elements =
      GraphParser::get_named_tensor_elements_or_throw(
          prefill_graph_info.raw_inputs, "attention_mask");

  prefill_attention_mask_columns =
      GraphParser::get_tensor_info_or_throw(prefill_graph_info.raw_inputs,
                                            "attention_mask")
    .dimensions.back(); // 8192

  prefill_sliding_attention_mask_elements = GraphParser::get_named_tensor_elements_or_throw (
      prefill_graph_info.raw_inputs, "sliding_attention_mask");

  prefill_sliding_attention_mask_columns
      = GraphParser::get_tensor_info_or_throw (prefill_graph_info.raw_inputs,
          "sliding_attention_mask")
            .dimensions.back (); // 768

  generation_attention_mask_elements = GraphParser::get_named_tensor_elements_or_throw (
      generation_graph_info.raw_inputs, "attention_mask");
  generation_sliding_attention_mask_elements = GraphParser::get_named_tensor_elements_or_throw (
      generation_graph_info.raw_inputs, "sliding_attention_mask");

  generation_full_kv_past_length = generation_attention_mask_elements - 1;
  generation_sliding_kv_past_length = generation_sliding_attention_mask_elements - 1;

  pos_dim = GraphParser::get_tensor_info_or_throw(
      prefill_graph_info.raw_inputs, "position_ids_cos").dimensions.back();
  swa_pos_dim = GraphParser::get_tensor_info_or_throw(
      prefill_graph_info.raw_inputs, "swa_position_ids_cos").dimensions.back();

  // ── Debug: confirm prefill and generation share the same pos/swa dims ──
  {
    const int gen_pos = GraphParser::get_tensor_info_or_throw(
        generation_graph_info.raw_inputs, "position_ids_cos")
        .dimensions.back();
    const int gen_swa = GraphParser::get_tensor_info_or_throw(
        generation_graph_info.raw_inputs, "swa_position_ids_cos")
        .dimensions.back();
    std::cout << "[ROPE-DBG] prefill pos_dim=" << pos_dim
              << " gen pos_dim=" << gen_pos
              << " | prefill swa_pos_dim=" << swa_pos_dim
              << " gen swa_pos_dim=" << gen_swa << std::endl;
    if (gen_pos != pos_dim || gen_swa != swa_pos_dim)
      std::cout << "[ROPE-DBG] !!! prefill/gen position dims DIFFER !!!"
                << std::endl;
  }

  rope_cache_seq_len = std::max(max_seq_len, generation_attention_mask_elements);

  // ── Bind input tensor pointers ──
  int prefill_input_idx    = GraphParser::find_tensor_index(
      prefill_graph_info.raw_inputs, "input_embeds");
  int generation_input_idx = GraphParser::find_tensor_index(
      generation_graph_info.raw_inputs, "input_embeds");
  input_sample      = std::get<float *>(prefill_inputs[prefill_input_idx]);
  generation_sample = std::get<float *>(generation_inputs[generation_input_idx]);

  attention_mask = std::get<uint16_t *>(prefill_inputs[
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs,
                                     "attention_mask")]);
  sliding_attention_mask = std::get<uint16_t *>(prefill_inputs[
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs,
                                     "sliding_attention_mask")]);
  generation_attention_mask = std::get<uint16_t *>(generation_inputs[
      GraphParser::find_tensor_index(generation_graph_info.raw_inputs,
                                     "attention_mask")]);
  generation_sliding_attention_mask = std::get<uint16_t *>(generation_inputs[
      GraphParser::find_tensor_index(generation_graph_info.raw_inputs,
                                     "sliding_attention_mask")]);

  prefill_position_ids_cos = std::get<uint16_t *>(prefill_inputs[
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs,
                                     "position_ids_cos")]);
  prefill_position_ids_sin = std::get<uint16_t *>(prefill_inputs[
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs,
                                     "position_ids_sin")]);
  generation_position_ids_cos = std::get<uint16_t *>(generation_inputs[
      GraphParser::find_tensor_index(generation_graph_info.raw_inputs,
                                     "position_ids_cos")]);
  generation_position_ids_sin = std::get<uint16_t *>(generation_inputs[
      GraphParser::find_tensor_index(generation_graph_info.raw_inputs,
                                     "position_ids_sin")]);
  prefill_swa_position_ids_cos = std::get<uint16_t *>(prefill_inputs[
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs,
                                     "swa_position_ids_cos")]);
  prefill_swa_position_ids_sin = std::get<uint16_t *>(prefill_inputs[
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs,
                                     "swa_position_ids_sin")]);
  generation_swa_position_ids_cos = std::get<uint16_t *>(generation_inputs[
      GraphParser::find_tensor_index(generation_graph_info.raw_inputs,
                                     "swa_position_ids_cos")]);
  generation_swa_position_ids_sin = std::get<uint16_t *>(generation_inputs[
      GraphParser::find_tensor_index(generation_graph_info.raw_inputs,
                                     "swa_position_ids_sin")]);

  // ── RoPE cache ──
  
  // std::tuple<uint16_t *, uint16_t *> cos_sin_tuple =
  //     get_cos_sin(rope_cache_seq_len, pos_dim, rope_theta);
  // position_ids_cos = std::get<0>(cos_sin_tuple);
  // position_ids_sin = std::get<1>(cos_sin_tuple);
  // allocated_ptrs_.insert(position_ids_cos);
  // allocated_ptrs_.insert(position_ids_sin);

  // std::tuple<uint16_t *, uint16_t *> swa_cos_sin_tuple =
  //     get_cos_sin(rope_cache_seq_len, swa_pos_dim, local_rope_theta);
  // swa_position_ids_cos = std::get<0>(swa_cos_sin_tuple);
  // swa_position_ids_sin = std::get<1>(swa_cos_sin_tuple);
  // allocated_ptrs_.insert(swa_position_ids_cos);
  // allocated_ptrs_.insert(swa_position_ids_sin);

  // ── Full attention RoPE ──
  double rope_scaling_factor_full = 1.0;

  std::tuple<uint16_t *, uint16_t *> cos_sin_tuple
      = get_cos_sin (rope_cache_seq_len, pos_dim, rope_theta_full,
          rope_type_full, rope_partial_factor, rope_scaling_factor_full);
  position_ids_cos = std::get<0> (cos_sin_tuple);
  position_ids_sin = std::get<1> (cos_sin_tuple);
  allocated_ptrs_.insert (position_ids_cos);
  allocated_ptrs_.insert (position_ids_sin);

  // ── Sliding window RoPE (default = no scaling) ──
  // HF Gemma 3/4 share a single RotaryEmbedding instance between full and
  // sliding attention, so the partial_rotary_factor read from
  // rope_parameters.full_attention applies to BOTH paths. Setting it to
  // 1.0 here would over-rotate the non-rotary lanes of the sliding head
  // and accumulate noise across layers, producing structured-but-garbled
  // output as the SWA path dominates short-range attention.
  std::tuple<uint16_t *, uint16_t *> swa_cos_sin_tuple
      = get_cos_sin (rope_cache_seq_len, swa_pos_dim, rope_theta_sliding,
          rope_type_sliding, rope_partial_factor, /*scaling=*/1.0);
  swa_position_ids_cos = std::get<0> (swa_cos_sin_tuple);
  swa_position_ids_sin = std::get<1> (swa_cos_sin_tuple);
  allocated_ptrs_.insert (swa_position_ids_cos);
  allocated_ptrs_.insert (swa_position_ids_sin);

  // ── PLE per-layer dst + scale/offset collection ──
  auto collect_per_layer = [] (const GraphInfo &gi,
                               std::vector<ml::train::TensorDim::IO_TensorType> &inputs,
                               std::vector<uint16_t *> &dsts,
                               std::vector<float> &scales, std::vector<int> &offsets,
                               std::vector<int> &model_indices) {
    std::map<int, std::tuple<uint16_t *, float, int>> by_index;
    for (size_t idx = 0; idx < gi.raw_inputs.size(); ++idx) {
      const auto &[name, info] = gi.raw_inputs[idx];
      const std::string prefix = "per_layer_inputs_";
      if (name.rfind(prefix, 0) != 0)
        continue;
      int n = std::stoi(name.substr(prefix.size()));
      by_index[n] = std::make_tuple(
          std::get<uint16_t *>(inputs[idx]), info.scale, info.offset);
    }
    dsts.clear();
    scales.clear ();
    offsets.clear ();
    model_indices.clear();
    for (auto &kv : by_index) {
      model_indices.push_back(kv.first);
      dsts.push_back(std::get<0>(kv.second));
      scales.push_back(std::get<1>(kv.second));
      offsets.push_back(std::get<2>(kv.second));
    }
  };
  collect_per_layer(prefill_graph_info, prefill_inputs,
                    prefill_per_layer_dst_, prefill_per_layer_scale_,
                    prefill_per_layer_offset_,
                    prefill_per_layer_model_index_);
  collect_per_layer(generation_graph_info, generation_inputs,
                    generation_per_layer_dst_, generation_per_layer_scale_,
                    generation_per_layer_offset_,
                    generation_per_layer_model_index_);

  std::cout << "[PLE] prefill slots=" << prefill_per_layer_dst_.size()
            << " generation slots=" << generation_per_layer_dst_.size()
            << std::endl;
  std::cout << "[PLE] prefill model indices: ";
  for (int n : prefill_per_layer_model_index_) std::cout << n << " ";
  std::cout << "\n[PLE] generation model indices: ";
  for (int n : generation_per_layer_model_index_) std::cout << n << " ";
  std::cout << std::endl;

  open_ple_file_();

  // ── KV cache mapping (Gauss 3.6 pattern, 2 KV per layer) ──
  this->kvs.clear();
  this->fresh_kvs.clear();
  this->kv_sizes.clear();
  this->kv_row_lengths.clear();
  this->kv_columns.clear();
  this->prefill_kvs.clear();
  this->prefill_kv_sizes.clear();
  this->prefill_kv_row_lengths.clear();
  this->prefill_to_generation_kv_indices.clear();
  this->prefill_kv_is_key.clear();
  this->prefill_output_kv_bindings.clear();
  this->generation_output_kv_bindings.clear();

  std::unordered_map<std::string, int> generation_kv_index_by_name;

  int kv_layer_count = 0;
  while(true){
    const std::string name ="past_key_"+std::to_string(kv_layer_count)+"_h0_in";
    if(find_tensor_index_or_minus_one(generation_graph_info.raw_inputs, name)<0)
      break;
    kv_layer_count++;
  }

  LOGD("KV layer count = %d (num_hidden_layers config = %d)", kv_layer_count, num_hidden_layers);
  for(int layer=0; layer<kv_layer_count;++layer){
    const std::vector<std::string> kv_names = {
      "past_key_"+std::to_string(layer)+"_h0_in",
      "past_value_"+std::to_string(layer)+"_h0_in",
    };

    {
      const auto &gen_key_info = GraphParser::get_tensor_info_or_throw(
          generation_graph_info.raw_inputs, kv_names[0]);
      this->kv_row_lengths.push_back(gen_key_info.dimensions.back());
      this->kv_columns.push_back(gen_key_info.dimensions[2]);

      // ── Debug: dump KV tensor shape for first 3 + last layer ──
      if (layer < 3 || layer == 34) {
        std::cout << "[KV-DBG] layer " << layer << " key dims=[";
        for (size_t i = 0; i < gen_key_info.dimensions.size(); ++i) {
          if (i) std::cout << ",";
          std::cout << gen_key_info.dimensions[i];
        }
        std::cout << "] → row_len=" << gen_key_info.dimensions.back()
                  << " columns(dim[2])=" << gen_key_info.dimensions[2]
                  << "\n";
        const auto &gen_val_info = GraphParser::get_tensor_info_or_throw(
            generation_graph_info.raw_inputs, kv_names[1]);
        std::cout << "[KV-DBG] layer " << layer << " val dims=[";
        for (size_t i = 0; i < gen_val_info.dimensions.size(); ++i) {
          if (i) std::cout << ",";
          std::cout << gen_val_info.dimensions[i];
        }
        std::cout << "]\n";
      }
    }

    for (const auto &name : kv_names) {
      int gen_idx = GraphParser::find_tensor_index (generation_graph_info.raw_inputs, name);
      int pre_idx = find_tensor_index_or_minus_one (prefill_graph_info.raw_inputs, name);
      const auto &gen_info = generation_graph_info.raw_inputs[gen_idx].second;
      int size = GraphParser::get_tensor_size (gen_info);

      // Layer-specific zero-point byte.
      int zq = std::max (0, std::min (255, -gen_info.offset));

      // ★ REUSE existing generation_inputs buffer (don't re-allocate).
      auto *current_kv = std::get<uint8_t *> (generation_inputs[gen_idx]);
      std::memset (current_kv, zq, size);

      // Separate fresh_kv to use as run-start reset state.
      auto *fresh_kv = static_cast<uint8_t *> (tracked_allocate (size));
      std::memset (fresh_kv, zq, size);
      allocated_ptrs_.insert (fresh_kv);

      int kv_input_index = (int)this->kvs.size ();
      this->kvs.push_back ((uint16_t *)current_kv);
      this->fresh_kvs.push_back ((uint16_t *)fresh_kv);
      this->kv_sizes.push_back (size);
      generation_kv_index_by_name[name] = kv_input_index;

      if (pre_idx >= 0) {
        const auto &pre_info = prefill_graph_info.raw_inputs[pre_idx].second;
        const bool is_key = starts_with (name, "past_key_");
        const int pre_row_length
            = is_key ? pre_info.dimensions.back () :
                       pre_info.dimensions[pre_info.dimensions.size () - 2];

        this->prefill_kvs.push_back (std::get<uint8_t *> (prefill_inputs[pre_idx]));
        this->prefill_kv_sizes.push_back (GraphParser::get_tensor_size (pre_info));
        this->prefill_kv_row_lengths.push_back (pre_row_length);
        this->prefill_to_generation_kv_indices.push_back (kv_input_index);
        this->prefill_kv_is_key.push_back (is_key ? 1 : 0);

        // Layer-specific zero point for prefill reset (replaces the 128 fill).
        int pzq = std::max (0, std::min (255, -pre_info.offset));
        this->prefill_kv_zero_byte_.push_back ((uint8_t)pzq);

        // Initialize prefill buffer to its layer-specific zero point.
        std::memset (std::get<uint8_t *> (prefill_inputs[pre_idx]), pzq,
            GraphParser::get_tensor_size (pre_info));
      }
    }
  }

  auto build_bindings = [&](const TensorInfoList &outs,
                            const std::string &graph_name) {
    std::vector<KvOutputBinding> bindings;
    for (size_t idx = 0; idx < outs.size(); ++idx) {
      const auto &name = outs[idx].first;
      if (!starts_with(name, "past_")) continue;
      auto in_name = kv_output_to_input_name(name);
      auto it = generation_kv_index_by_name.find(in_name);
      if (it == generation_kv_index_by_name.end())
        throw std::runtime_error(graph_name +
            " KV output has no matching generation input: " + name);
      int kv_index = it->second;
      bindings.push_back({(int)idx, kv_index, kv_index / 2,
                          starts_with(name, "past_key_")});
    }
    return bindings;
  };
  prefill_output_kv_bindings    = build_bindings(prefill_graph_info.raw_outputs,
                                                  prefill_graph);
  generation_output_kv_bindings = build_bindings(generation_graph_info.raw_outputs,
                                                  generation_graph);

  LOGD("KV mapping: gen_inputs=%zu pre_inputs=%zu pre_outs=%zu gen_outs=%zu",
       this->kvs.size(), this->prefill_kvs.size(),
       this->prefill_output_kv_bindings.size(),
       this->generation_output_kv_bindings.size());

  // ── Logit dequant params (overrides setupParameters defaults) ──
  const auto &logits_info = GraphParser::get_tensor_info_or_throw(
      generation_graph_info.raw_outputs, "logits");
  logit_scale  = logits_info.scale;
  logit_offset = logits_info.offset;

  initialize_kv_cache();

  LOGD("----------------------- initialize() done");
}

// =====================================================================
// setupParameters
// =====================================================================
void Gemma4_E2B_QNN::setupParameters(json &cfg, json &generation_cfg,
                                     json &nntr_cfg) {
  Quick_Dot_AI_QNN::setupParameters(cfg, generation_cfg, nntr_cfg);

  num_hidden_layers = cfg["num_hidden_layers"].get<int>();
  hidden_size       = cfg["hidden_size"].get<int>();
  vocab_size        = cfg["vocab_size"].get<int>();
  max_seq_len       = cfg["max_seq_len"].get<int>();
  sliding_window    = cfg["sliding_window"].get<int>();
  context_size      = cfg["context_size"].get<int>();
  g_head_dim        = cfg["global_head_dim"].get<int>();
  l_head_dim        = cfg["head_dim"].get<int>();
  head_dim          = g_head_dim;

  // ─── RoPE parameters (new format preferred) ───
  if (cfg.contains("rope_parameters") && cfg["rope_parameters"].is_object()) {
    auto &rp = cfg["rope_parameters"];

    if (rp.contains("full_attention") && rp["full_attention"].is_object()) {
      auto &fa = rp["full_attention"];
      rope_theta_full     = fa.value("rope_theta", 1000000.0f);
      rope_partial_factor = fa.value("partial_rotary_factor", 1.0f);
      rope_type_full      = fa.value("rope_type", std::string("default"));
    }

    if (rp.contains("sliding_attention") &&
        rp["sliding_attention"].is_object()) {
      auto &sa = rp["sliding_attention"];
      rope_theta_sliding = sa.value("rope_theta", 10000.0f);
      rope_type_sliding  = sa.value("rope_type", std::string("default"));
    }
  } else {
    // Legacy flat form
    rope_theta_full    = cfg.value("rope_theta",       1000000.0f);
    rope_theta_sliding = cfg.value("local_rope_theta",   10000.0f);
  }

  // rope_theta       = rope_theta_full;
  // local_rope_theta = rope_theta_sliding;

  LOGD("RoPE full: theta=%f partial=%f type=%s",
       rope_theta_full, rope_partial_factor, rope_type_full.c_str());
  LOGD("RoPE sliding: theta=%f type=%s",
       rope_theta_sliding, rope_type_sliding.c_str());

  padding_token      = generation_cfg["pad_token_id"].get<int>();
  eos_tokens         = generation_cfg["eos_token_id"].get<std::vector<int>>();
  temperature        = generation_cfg["temperature"].get<float>();
  top_k              = generation_cfg["top_k"].get<int>();
  top_p              = generation_cfg["top_p"].get<float>();
  repetition_penalty = generation_cfg.value("repetition_penalty", 1.0f);
  logit_scale        = generation_cfg.value("logit_scale", 1.0f);
  logit_offset       = generation_cfg.value("logit_offset", 0);

  // Gemma final-logit soft-cap (0 disables). Without it the model
  // collapses into repetition since a few raw logits dominate softmax.
  final_logit_softcapping = cfg.value("final_logit_softcapping", 0.0f);
  LOGD("final_logit_softcapping = %f", final_logit_softcapping);

  lora_path     = nntr_cfg.value("lora_path", "");
  lora_path     = rebase_relative_to_model_file(lora_path, model_file_name);
  ple_file_name = nntr_cfg.value("ple_file_name", "");
  ple_file_name = rebase_relative_to_model_file(ple_file_name, model_file_name);
}

// =====================================================================
// run()
// =====================================================================
void Gemma4_E2B_QNN::run(const WSTR prompt, bool /*do_sample*/,
                        const WSTR /*system_prompt*/, const WSTR /*tail_prompt*/,
                        bool log_output) {
  last_output_.clear();
  stop_requested_.store(false, std::memory_order_release);

  std::string prefill_graph    = graphs_to_use[0];
  std::string generation_graph = graphs_to_use[1];
  auto &prefill_inputs    = models[prefill_graph].model_inputs;
  auto &generation_inputs = models[generation_graph].model_inputs;
  auto &prefill_model     = models[prefill_graph].model_handle;
  auto &generation_model  = models[generation_graph].model_handle;

  auto _input = tokenizer->Encode(prompt);
  if (_input.size() <= 1) {
    std::cout << "[Error] Empty input\n";
    return;
  }
  unsigned int input_len = _input.size() - 1;
  int          token     = _input.back();
  auto n_chunks = (input_len % 256 != 0)
                  ? ((input_len / 256) + 1) : (input_len / 256);

  if (kv_len + (int)input_len >= generation_full_kv_past_length)
    throw std::runtime_error("Input prompt leaves no room for generation");

  std::vector<int> output;
  std::vector<ml::train::TensorDim::IO_TensorType> outputs;

  // ── Lambdas (Gauss 3.6 style) ──
  auto fill_generation_inputs = [&](int current_token, int position) {
    if (position < 0 || position >= rope_cache_seq_len)
      throw std::runtime_error("Generation position out of rope cache");

    generation_sample[0] = current_token;

    std::fill_n(generation_attention_mask,
                generation_attention_mask_elements, 0);
    std::fill_n(generation_sliding_attention_mask,
                generation_sliding_attention_mask_elements, 0);
    generation_attention_mask[generation_attention_mask_elements - 1] =
        std::numeric_limits<uint16_t>::max();
    generation_sliding_attention_mask[generation_sliding_attention_mask_elements - 1] =
        std::numeric_limits<uint16_t>::max();

    for (int i = 0; i < position && i < generation_full_kv_past_length; ++i)
      generation_attention_mask[i] = std::numeric_limits<uint16_t>::max();
    for (int i = 0; i < position && i < generation_sliding_kv_past_length; ++i)
      generation_sliding_attention_mask[i] = std::numeric_limits<uint16_t>::max();

    std::memcpy(generation_position_ids_cos,
                position_ids_cos + position * pos_dim,
                pos_dim * sizeof(uint16_t));
    std::memcpy(generation_position_ids_sin,
                position_ids_sin + position * pos_dim,
                pos_dim * sizeof(uint16_t));
    std::memcpy(generation_swa_position_ids_cos,
                swa_position_ids_cos + position * swa_pos_dim,
                swa_pos_dim * sizeof(uint16_t));
    std::memcpy(generation_swa_position_ids_sin,
                swa_position_ids_sin + position * swa_pos_dim,
                swa_pos_dim * sizeof(uint16_t));
  };

  auto append_outputs_to_kv_cache = [&](
      const std::vector<ml::train::TensorDim::IO_TensorType> &step_outputs,
      const std::vector<KvOutputBinding> &bindings,
      int target_position, int rows, int src_row_length,
      const std::string &graph_name) {

    for (const auto &b : bindings) {
      if (b.output_index < 0 || b.output_index >= (int)step_outputs.size() ||
          b.kv_index < 0 || b.kv_index >= (int)kvs.size() ||
          b.layer_index < 0 || b.layer_index >= (int)kv_row_lengths.size())
        throw std::runtime_error(graph_name + " KV binding out of range");
    }

#pragma omp parallel for
    for (int bi = 0; bi < (int)bindings.size(); ++bi) {
      const auto &b   = bindings[bi];
      int dest_row_length = kv_row_lengths[b.layer_index];
      int num_column      = kv_columns[b.layer_index];
      auto out  = std::get<uint8_t *>(step_outputs[b.output_index]);
      auto dest = (uint8_t *)kvs[b.kv_index];

      int target_idx = target_position;
      int valid_before = std::min(target_position, dest_row_length);
      int shift = valid_before + rows - dest_row_length;
      if (shift > 0) {
        target_idx = valid_before - shift;
        if (b.is_key) {
          for (int col = 0; col < num_column; ++col) {
            uint8_t *col_base = dest + col * dest_row_length;
            std::memmove(col_base, col_base + shift, dest_row_length - shift);
          }
        } else {
          std::memmove(dest, dest + shift * num_column,
                       (dest_row_length - shift) * num_column);
        }
      }

      if (b.is_key) {
        process_key(out, rows, num_column, dest, target_idx,
                    dest_row_length, src_row_length);
      } else {
        process_value(out, rows, num_column, dest, target_idx);
      }
    }
  };

  auto append_generation_token_to_kv_cache = [&](int t) {
    if (kv_len >= generation_full_kv_past_length) return;
    fill_generation_inputs(t, kv_len);
    fill_generation_ple_(t);
    auto term = generation_model->inference(1, generation_inputs);
    append_outputs_to_kv_cache(term, generation_output_kv_bindings,
                               kv_len, 1, 1, generation_graph);
    kv_len += 1;
  };

  // ── Prefill ──
  for (int c = 0; c < (int)n_chunks; ++c) {
    int chunk_len = ((c + 1) * 256 < (int)input_len)
                    ? context_size : ((int)input_len - c * 256);

    sync_generation_kv_cache_to_prefill();

    for (int i = 0; i < context_size; ++i)
      input_sample[i] = (i < chunk_len) ? _input[c * 256 + i] : padding_token;

    fill_attention_mask_with_length(context_size, prefill_attention_mask_columns,
                                    chunk_len, attention_mask);
    // Past KV cap MUST reserve the trailing `context_size` cols for the
    // current chunk's causal triangle (those cols are written by
    // fill_attention_mask_with_length above). Using
    // generation_full_kv_past_length here would overlap chunk cols when
    // kv_len > prefill_attention_mask_columns - context_size, corrupting
    // the mask on multi-chunk prefill. Match Gauss 3.6 convention:
    //   past_max = mask_columns - context_size
    {
      const int prefill_full_past_max =
          prefill_attention_mask_columns - context_size;
      fill_attention_mask_with_prev_length(
          context_size, prefill_attention_mask_columns,
          std::min(kv_len, prefill_full_past_max), attention_mask);
    }
    fill_attention_mask_with_length(context_size,
                                    prefill_sliding_attention_mask_columns,
                                    chunk_len, sliding_attention_mask);
    {
      const int prefill_sliding_past_max =
          prefill_sliding_attention_mask_columns - context_size;
      fill_attention_mask_with_prev_length(
          context_size, prefill_sliding_attention_mask_columns,
          std::min(kv_len, prefill_sliding_past_max),
          sliding_attention_mask);
    }

    std::fill_n(prefill_position_ids_cos, context_size * pos_dim, 65535);
    std::fill_n(prefill_position_ids_sin, context_size * pos_dim, 32768);
    std::fill_n(prefill_swa_position_ids_cos, context_size * swa_pos_dim, 65535);
    std::fill_n(prefill_swa_position_ids_sin, context_size * swa_pos_dim, 32768);

    if (kv_len + chunk_len > rope_cache_seq_len)
      throw std::runtime_error("Prefill position out of rope cache");

    auto pos_off     = kv_len * pos_dim;
    auto swa_pos_off = kv_len * swa_pos_dim;
    std::memcpy(prefill_position_ids_cos, position_ids_cos + pos_off,
                chunk_len * pos_dim * sizeof(uint16_t));
    std::memcpy(prefill_position_ids_sin, position_ids_sin + pos_off,
                chunk_len * pos_dim * sizeof(uint16_t));
    std::memcpy(prefill_swa_position_ids_cos, swa_position_ids_cos + swa_pos_off,
                chunk_len * swa_pos_dim * sizeof(uint16_t));
    std::memcpy(prefill_swa_position_ids_sin, swa_position_ids_sin + swa_pos_off,
                chunk_len * swa_pos_dim * sizeof(uint16_t));

    fill_prefill_ple_chunk_(_input, c, chunk_len);

    outputs = prefill_model->inference(1, prefill_inputs);
    append_outputs_to_kv_cache(outputs, prefill_output_kv_bindings,
                               kv_len, chunk_len, context_size, prefill_graph);
    kv_len += chunk_len;
  }

  // ── Generation ──
  auto start = std::chrono::system_clock::now();
  int idx;
  int prefill_len = kv_len;
  for (idx = prefill_len; idx < generation_full_kv_past_length; ++idx) {
    fill_generation_inputs(token, idx);
    fill_generation_ple_(token);
    outputs = generation_model->inference(1, generation_inputs);
    append_outputs_to_kv_cache(outputs, generation_output_kv_bindings,
                               idx, 1, 1, generation_graph);
    kv_len += 1;
    token = sample(std::get<uint16_t *>(outputs.back()), vocab_size,
                   _input.data(), _input.size(), logit_scale, logit_offset,
                   repetition_penalty, temperature, top_p, top_k,
                   final_logit_softcapping);
    output.push_back(token);

    bool reached_eos = false;
    for (auto eos : eos_tokens) {
      if (token == eos) {
	reached_eos = true;
	break;
      }
    }
    if (reached_eos) {
      append_generation_token_to_kv_cache(token);
      break;
    }

    std::string decoded = tokenizer->Decode({ token });
    last_output_ += decoded;
    LOGD("%d : %s", token, decoded.c_str());
    if (streamer_) {
      if (streamer_put(streamer_, decoded.c_str()) != 0) {
        stop_requested_.store(true, std::memory_order_release);
        break;
      }
    } else if (log_output) {
      std::cout << decoded << std::flush;
    }
    _input.push_back(token);

    if (stop_requested_.load(std::memory_order_acquire)) break;
  }

  if (streamer_) streamer_end(streamer_);
  has_run_ = true;
  conversation_started_ = true;

  auto end = std::chrono::system_clock::now();
  raw_exec_seconds = end - start;
  if (log_output) {
    std::cout << "\n\nGeneration exec_time : " << raw_exec_seconds.count()
              << ", token per second: "
              << (idx - prefill_len) / raw_exec_seconds.count()
              << ", token generation time average: "
              << raw_exec_seconds.count() / std::max(1, idx - prefill_len)
              << std::endl;
  }
}

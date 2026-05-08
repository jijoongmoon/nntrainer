// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gemma4_e2b_qnn.cpp
 * @brief  QNN model implementation with self-registration
 * @note   This model auto-registers with the CausalLM Factory via
 * __attribute__((constructor)) when linked or loaded.
 *
 *         No modification to nntrainer's main.cpp is needed.
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
#include <cstring>
#include <iostream>
#include <limits>
#include <utility>
#include "json.hpp"
#include <fstream>
#include <map>
#include <algorithm>

using namespace causallm;

namespace
{

// LUT (lut_scale, lut_offset)
// QNN consumer (out_scale, out_offset) convert UINT16
//   f      = (q4bit + lut_offset) * lut_scale
//   q16bit = round(f / out_scale - out_offset)
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


bool
is_absolute_path (const std::string &path)
{
  return !path.empty () && path[0] == '/';
}

std::string dirname(const std::string &path) {
  auto pos = path.find_last_of('/');
  if (pos == std::string::npos) {
    return "";
  }
  return path.substr(0, pos);
}

std::string rebase_relative_to_model_file(const std::string &path,
                                          const std::string &model_file) {
  if (path.empty() || is_absolute_path(path)) {
    return path;
  }

  auto base_dir = dirname(model_file);
  if (base_dir.empty()) {
    return path;
  }
  return base_dir + "/" + path;
}

} // namespace

/**
 * @brief Auto-registration via constructor attribute
 *
 * This function runs automatically when the shared library is loaded
 * (before main()). It registers all custom models with the CausalLM Factory.
 *
 */
__attribute__((constructor)) static void register_custom_models() {
  causallm::Factory::Instance().registerModel(
      "Gemma4_E2B_QNN", [](causallm::json cfg, causallm::json generation_cfg,
                          causallm::json nntr_cfg) {
        return std::make_unique<causallm::Gemma4_E2B_QNN>(cfg, generation_cfg,
                                                        nntr_cfg);
      });
}

static void copy_ple_inputs_from_file(
    const std::string &ple_file_name,
    const std::vector<std::pair<const GraphInfo *,
                                std::vector<ml::train::TensorDim::IO_TensorType> *>>
        &targets) {
  if (ple_file_name.empty()) {
    return;
  }

  int fd = open(ple_file_name.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("Failed to open ple_file_name: " + ple_file_name);
  }

  struct stat st;
  if (fstat(fd, &st) < 0) {
    close(fd);
    throw std::runtime_error("Failed to stat ple_file_name: " + ple_file_name);
  }

  size_t file_size = st.st_size;
  void *mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED) {
    close(fd);
    throw std::runtime_error("Failed to mmap ple_file_name: " + ple_file_name);
  }

  auto *data_ptr = static_cast<uint8_t *>(mapped);
  size_t remaining = file_size;

  for (const auto &[graph_info, inputs] : targets) {
    std::cout << "ple : "<< ple_file_name<< " " <<graph_info->raw_inputs.size() << std::endl;
    for (size_t idx = 0; idx < graph_info->raw_inputs.size(); idx++) {
      const auto &[name, info] = graph_info->raw_inputs[idx];
      if (name.find("per_layer_inputs_") != 0) {
        continue;
      }

      int size = GraphParser::get_tensor_size(info);
      if (remaining < static_cast<size_t>(size)) {
        munmap(mapped, file_size);
        close(fd);
        throw std::runtime_error("PLE file is smaller than per_layer_inputs");
      }

      std::memcpy(std::get<uint16_t *>((*inputs)[idx]), data_ptr, size);
      data_ptr += size;
      remaining -= size;
    }
  }

  munmap(mapped, file_size);
  close(fd);
}

void Gemma4_E2B_QNN::open_ple_file_(){
  if(ple_file_name.empty())return;
  std::ifstream mf(ple_file_name);
  if(!mf.is_open())
    throw std::runtime_error("Failed to open PLE mainfest: " + ple_file_name);
  json j; mf >>j;

  const std::string lut_rel = j.at("lut-path").get<std::string>();
  const int row_elems = j.at("size").get<int>();
  const std::string datatype= j.value("datatype", std::string("ufixed8"));
  const auto &qp = j.at("quant-param");

  if(datatype!= "ufixed8")
    throw std::runtime_error("PLE: only ufixed8 suppored, got " + datatype);

  ple_scale_ = qp.at("scale").get<float>();
  ple_offset_= qp.at("offset").get<int>();
  ple_row_elems_ = static_cast<size_t>(row_elems);
  ple_row_bytes_ = (ple_row_elems_ + 1)/2;
  ple_per_layer_ = 256;
  ple_layers_ = ple_row_elems_ / ple_per_layer_;

  if(ple_layers_ * ple_per_layer_ != ple_row_elems_)
    throw std::runtime_error("PLE 'size' not divisible by 256");

  if(generation_per_layer_dst_.size() > ple_layers_)
    throw std::runtime_error("PLE layer count too small");

  std::string lut_abs = rebase_relative_to_model_file(lut_rel, ple_file_name);

  ple_fd_ = open(lut_abs.c_str(), O_RDONLY);
  if(ple_fd_<0) throw std::runtime_error("open PLE bin: "+lut_abs);
  struct stat st;
  if(fstat(ple_fd_, &st)<0){
    ::close(ple_fd_);
    ple_fd_ = -1;
    throw std::runtime_error("stat PLE bin: "+lut_abs);
  }
  ple_file_size_ = static_cast<size_t>(st.st_size);
  if(ple_file_size_ % ple_row_bytes_ !=0){
    ::close(ple_fd_); ple_fd_=-1;
    throw std::runtime_error("PLE bin size not multiple of row bytes");
  }

  void *m = mmap(nullptr, ple_file_size_, PROT_READ, MAP_PRIVATE, ple_fd_, 0);
  if(m==MAP_FAILED){
    ::close(ple_fd_);ple_fd_=-1;
    throw std::runtime_error("mmap PLE bin: "+ lut_abs);
  }
  ple_mmap_ = static_cast<const uint8_t*>(m);
#ifdef POSIX_MADV_RANDOM
  posix_madvise((void*)ple_mmap_, ple_file_size_, POSIX_MADV_RANDOM);
#endif
  std::cout << "[PLE] mmaped "<<lut_abs
	    <<" rows="<<(ple_file_size_ / ple_row_bytes_)
	    <<" scale = "<<ple_scale_ << " offset= "<<ple_offset_<<std::endl;
}

void
Gemma4_E2B_QNN::close_ple_file_()
{
  if (ple_mmap_) {
    munmap ((void *)ple_mmap_, ple_file_size_);
    ple_mmap_ = nullptr;
  }
  if (ple_fd_ >= 0) {
    ::close (ple_fd_);
    ple_fd_ = -1;
  }
}

void
Gemma4_E2B_QNN::fill_prefill_ple_chunk_ (
    const std::vector<int> &tokens, int chunk_idx, int chunk_len)
{
  if (!ple_mmap_)
    return;
  const size_t L_pre = prefill_per_layer_dst_.size (); // 14
  const size_t per_layer_elems = ple_per_layer_; // 256
  const size_t per_layer_bytes = per_layer_elems / 2; // 128
  const int chunk_size_tokens = context_size; // 256

  for (int t = 0; t < chunk_size_tokens; ++t) {
    const int abs_idx = chunk_idx * chunk_size_tokens + t;
    const int token_id = (t < chunk_len) ? tokens[abs_idx] : padding_token;
    const uint8_t *row = ple_mmap_ + (size_t)token_id * ple_row_bytes_;
    for (size_t l = 0; l < L_pre; ++l) {
      uint16_t *dst = prefill_per_layer_dst_[l] + (size_t)t * per_layer_elems;
      dequant_nibbles_requant_u16 (row + l * per_layer_bytes, per_layer_elems, ple_scale_,
          ple_offset_, prefill_per_layer_scale_[l], prefill_per_layer_offset_[l], dst);
    }
  }
    static bool ple_dbg_done = false;
    if (!ple_dbg_done && chunk_idx == 0) {
      ple_dbg_done = true;

      const int abs_idx = 0;
      const int token_id = tokens[abs_idx]; // 실제 사용된 첫 토큰 id
      const uint8_t *row = ple_mmap_ + (size_t)token_id * ple_row_bytes_;

      std::cout << "[PLE-DBG] token_id=" << token_id
                << " row offset=" << (size_t)token_id * ple_row_bytes_
                << " ple_row_bytes_=" << ple_row_bytes_ << "\n";

      std::cout << "[PLE-DBG] L0 raw bytes [0..15]: ";
      for (int i = 0; i < 16; ++i)
        std::cout << std::hex << (int)row[i] << " ";
      std::cout << std::dec << "\n";

      std::cout << "[PLE-DBG] L0 nibbles  [0..31]: ";
      for (int i = 0; i < 16; ++i) {
        std::cout << (int)(row[i] & 0x0F) << " " << (int)((row[i] >> 4) & 0x0F) << " ";
      }
      std::cout << "\n";

      std::cout << "[PLE-DBG] L0 dst[0..31]: ";
      for (int i = 0; i < 32; ++i)
        std::cout << prefill_per_layer_dst_[0][i] << " ";
      std::cout << "\n";

      // 다른 layer 도 확인 (L13 = prefill 의 마지막 layer)
      std::cout << "[PLE-DBG] L13 dst[0..7]: ";
      for (int i = 0; i < 8; ++i)
        std::cout << prefill_per_layer_dst_[13][i] << " ";
      std::cout << "\n";
    }
}

void
Gemma4_E2B_QNN::fill_generation_ple_(int token_id)
{
  if (!ple_mmap_)
    return;
  const size_t L_gen = generation_per_layer_dst_.size (); // 35
  const size_t per_layer_elems = ple_per_layer_; // 256
  const size_t per_layer_bytes = per_layer_elems / 2; // 128
  const uint8_t *row = ple_mmap_ + (size_t)token_id * ple_row_bytes_;
  for (size_t l = 0; l < L_gen; ++l) {
    dequant_nibbles_requant_u16 (row + l * per_layer_bytes, per_layer_elems,
        ple_scale_, ple_offset_, generation_per_layer_scale_[l],
        generation_per_layer_offset_[l], generation_per_layer_dst_[l]);
  }
}


Gemma4_E2B_QNN::~Gemma4_E2B_QNN(){
  close_ple_file_();
}

void causallm::Gemma4_E2B_QNN::initialize() {
  Quick_Dot_AI_QNN::initialize();
  LOGD("Quick_Dot_AI_QNN::initialize() done");

  std::string prefill_graph = graphs_to_use[0];
  std::string generation_graph = graphs_to_use[1];

  LOGD("----------------------- initialize() %s, %s", prefill_graph.c_str(),
       generation_graph.c_str());
  auto &prefill_graph_info = models[prefill_graph].graph_info;
  auto &generation_graph_info = models[generation_graph].graph_info;
  auto &prefill_inputs = models[prefill_graph].model_inputs;
  auto &generation_inputs = models[generation_graph].model_inputs;

  std::cout << "prefill_inputs.size : "<<prefill_inputs.size()<<std::endl;
  std::cout << "generation_inputs.size : "<<generation_inputs.size()<<std::endl;

  if (prefill_inputs.size() != prefill_graph_info.raw_inputs.size()) {
    throw std::runtime_error("Gemma4 prefill input count mismatch: graph=" +
                             std::to_string(prefill_graph_info.raw_inputs.size()) +
                             ", model=" +
                             std::to_string(prefill_inputs.size()));
  }
  if (generation_inputs.size() != generation_graph_info.raw_inputs.size()) {
    throw std::runtime_error(
        "Gemma4 generation input count mismatch: graph=" +
        std::to_string(generation_graph_info.raw_inputs.size()) + ", model=" +
        std::to_string(generation_inputs.size()));
  }

  prefill_attention_mask_elements =
      GraphParser::get_named_tensor_elements_or_throw(
          prefill_graph_info.raw_inputs, "attention_mask");
  prefill_attention_mask_columns =
      GraphParser::get_tensor_info_or_throw(prefill_graph_info.raw_inputs,
                                            "attention_mask")
          .dimensions.back();
  prefill_sliding_attention_mask_elements =
      GraphParser::get_named_tensor_elements_or_throw(
          prefill_graph_info.raw_inputs, "sliding_attention_mask");
  prefill_sliding_attention_mask_columns =
      GraphParser::get_tensor_info_or_throw(prefill_graph_info.raw_inputs,
                                            "sliding_attention_mask")
          .dimensions.back();
  generation_attention_mask_elements =
      GraphParser::get_named_tensor_elements_or_throw(
          generation_graph_info.raw_inputs, "attention_mask");
  generation_sliding_attention_mask_elements =
      GraphParser::get_named_tensor_elements_or_throw(
          generation_graph_info.raw_inputs, "sliding_attention_mask");

  std::cout <<prefill_attention_mask_elements << " "<< prefill_attention_mask_columns << " " <<  prefill_sliding_attention_mask_elements << " "<<prefill_sliding_attention_mask_columns << " " << generation_attention_mask_elements << " " << generation_sliding_attention_mask_elements << std::endl;
  
  generation_full_kv_past_length = generation_attention_mask_elements - 1;
  generation_sliding_kv_past_length =
      generation_sliding_attention_mask_elements - 1;

  pos_dim = GraphParser::get_tensor_info_or_throw(prefill_graph_info.raw_inputs,
                                                 "position_ids_cos")
                .dimensions.back();
  swa_pos_dim =
      GraphParser::get_tensor_info_or_throw(prefill_graph_info.raw_inputs,
                                            "swa_position_ids_cos")
          .dimensions.back();
  
  rope_cache_seq_len = std::max(max_seq_len, generation_attention_mask_elements);
  std::cout <<"max_seq_len : "<< max_seq_len<< " " << generation_attention_mask_elements<<std::endl;

  int prefill_input_idx
      = GraphParser::find_tensor_index (prefill_graph_info.raw_inputs, "input_embeds");

  int generation_input_idx = GraphParser::find_tensor_index (
      generation_graph_info.raw_inputs, "input_embeds");
  
  LOGD("----------------------- prefill_inputs_idx : %d generation_input_idx :%d ", prefill_input_idx, generation_input_idx);
  
  input_sample = std::get<float *>(prefill_inputs[prefill_input_idx]);
  generation_sample = std::get<float *>(generation_inputs[generation_input_idx]);

  LOGD("----------------------- %d %f ",input_sample[0], generation_sample[0]);

  int prefill_attn_mask_idx =
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs, "attention_mask");
  int prefill_sliding_attn_mask_idx = GraphParser::find_tensor_index(
      prefill_graph_info.raw_inputs, "sliding_attention_mask");
  int generation_attn_mask_idx =
      GraphParser::find_tensor_index(generation_graph_info.raw_inputs, "attention_mask");
  int generation_sliding_attn_mask_idx = GraphParser::find_tensor_index(
      generation_graph_info.raw_inputs, "sliding_attention_mask");

  attention_mask = std::get<uint16_t *>(prefill_inputs[prefill_attn_mask_idx]);
  sliding_attention_mask =
      std::get<uint16_t *>(prefill_inputs[prefill_sliding_attn_mask_idx]);
  generation_attention_mask =
      std::get<uint16_t *>(generation_inputs[generation_attn_mask_idx]);
  generation_sliding_attention_mask =
      std::get<uint16_t *>(generation_inputs[generation_sliding_attn_mask_idx]);

  int prefill_pos_cos_idx =
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs, "position_ids_cos");
  int prefill_pos_sin_idx =
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs, "position_ids_sin");
  int generation_pos_cos_idx =
      GraphParser::find_tensor_index(generation_graph_info.raw_inputs, "position_ids_cos");
  int generation_pos_sin_idx =
      GraphParser::find_tensor_index(generation_graph_info.raw_inputs, "position_ids_sin");

  prefill_position_ids_cos =
      std::get<uint16_t *>(prefill_inputs[prefill_pos_cos_idx]);
  prefill_position_ids_sin =
      std::get<uint16_t *>(prefill_inputs[prefill_pos_sin_idx]);
  generation_position_ids_cos =
      std::get<uint16_t *>(generation_inputs[generation_pos_cos_idx]);
  generation_position_ids_sin =
      std::get<uint16_t *>(generation_inputs[generation_pos_sin_idx]);

  int prefill_swa_pos_cos_idx =
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs, "swa_position_ids_cos");
  int prefill_swa_pos_sin_idx =
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs, "swa_position_ids_sin");
  int generation_swa_pos_cos_idx = GraphParser::find_tensor_index(
      generation_graph_info.raw_inputs, "swa_position_ids_cos");
  int generation_swa_pos_sin_idx = GraphParser::find_tensor_index(
      generation_graph_info.raw_inputs, "swa_position_ids_sin");

  prefill_swa_position_ids_cos =
      std::get<uint16_t *>(prefill_inputs[prefill_swa_pos_cos_idx]);
  prefill_swa_position_ids_sin =
      std::get<uint16_t *>(prefill_inputs[prefill_swa_pos_sin_idx]);
  generation_swa_position_ids_cos =
      std::get<uint16_t *>(generation_inputs[generation_swa_pos_cos_idx]);
  generation_swa_position_ids_sin =
      std::get<uint16_t *>(generation_inputs[generation_swa_pos_sin_idx]);

  std::tuple<uint16_t *, uint16_t *> cos_sin_tuple =
      get_cos_sin(rope_cache_seq_len, pos_dim, rope_theta);
  position_ids_cos = std::get<0>(cos_sin_tuple);
  position_ids_sin = std::get<1>(cos_sin_tuple);
  allocated_ptrs_.insert(position_ids_cos);
  allocated_ptrs_.insert(position_ids_sin);

  std::tuple<uint16_t *, uint16_t *> swa_cos_sin_tuple =
      get_cos_sin(rope_cache_seq_len, swa_pos_dim, local_rope_theta);
  swa_position_ids_cos = std::get<0>(swa_cos_sin_tuple);
  swa_position_ids_sin = std::get<1>(swa_cos_sin_tuple);
  allocated_ptrs_.insert (swa_position_ids_cos);
  allocated_ptrs_.insert (swa_position_ids_sin);

  auto collect_per_layer
      = [] (const GraphInfo &gi, std::vector<ml::train::TensorDim::IO_TensorType> &inputs,
            std::vector<uint16_t *> &dsts, std::vector<float> &scales,
            std::vector<int> &offsets) {
          std::map<int, std::tuple<uint16_t *, float, int>> by_index;
          for (size_t idx = 0; idx < gi.raw_inputs.size (); ++idx) {
            const auto &[name, info] = gi.raw_inputs[idx];
            const std::string prefix = "per_layer_inputs_";
            if (name.rfind (prefix, 0) != 0)
              continue;
            int n = std::stoi (name.substr (prefix.size ()));
            by_index[n] = std::make_tuple (
                std::get<uint16_t *> (inputs[idx]), info.scale, info.offset);
          }
          dsts.clear ();
          scales.clear ();
          offsets.clear ();
          for (auto &kv : by_index) {
            dsts.push_back (std::get<0> (kv.second));
            scales.push_back (std::get<1> (kv.second));
            offsets.push_back (std::get<2> (kv.second));
          }
        };

  collect_per_layer (prefill_graph_info, prefill_inputs, prefill_per_layer_dst_,
      prefill_per_layer_scale_, prefill_per_layer_offset_);
  collect_per_layer (generation_graph_info, generation_inputs, generation_per_layer_dst_,
      generation_per_layer_scale_, generation_per_layer_offset_);


  std::cout << "[PLE] prefill slots=" << prefill_per_layer_dst_.size ()
            << " generation slots=" << generation_per_layer_dst_.size () << std::endl;

  open_ple_file_ ();


  // if (lora_path.empty ()) {
  //   for (size_t idx = 0; idx < generation_graph_info.raw_inputs.size (); idx++) {
  //     const auto &[name, info] = generation_graph_info.raw_inputs[idx];
  //     if (name.find ("_lora_") != std::string::npos) {
  //       int size = GraphParser::get_tensor_size (info);
  //       auto *lora_ptr = std::get<uint16_t *> (generation_inputs[idx]);
  //       std::fill_n (lora_ptr, size / sizeof (uint16_t), 32768);
  //     }
  //   }
  // } else {
  //   int fd = open (lora_path.c_str (), O_RDONLY);
  //   if (fd < 0) {
  //     throw std::runtime_error ("Failed to open lora_path: " + lora_path);
  //   }

  //   struct stat st;
  //   if (fstat (fd, &st) < 0) {
  //     close (fd);
  //     throw std::runtime_error ("Failed to stat lora_path: " + lora_path);
  //   }
  //   size_t file_size = st.st_size;

  //   void *mapped = mmap (nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  //   if (mapped == MAP_FAILED) {
  //     close (fd);
  //     throw std::runtime_error ("Failed to mmap lora_path: " + lora_path);
  //   }

  //   uint8_t *data_ptr = static_cast<uint8_t *> (mapped);

  //   for (size_t idx = 0; idx < prefill_graph_info.raw_inputs.size (); idx++) {
  //     const auto &[name, info] = prefill_graph_info.raw_inputs[idx];
  //     if (name.find ("_lora_") != std::string::npos) {
  //       int size = GraphParser::get_tensor_size (info);
  //       memcpy (std::get<uint16_t *> (prefill_inputs[idx]), data_ptr, size);
  //       data_ptr += size;
  //     }
  //   }
  //   for (size_t idx = 0; idx < generation_graph_info.raw_inputs.size (); idx++) {
  //     const auto &[name, info] = generation_graph_info.raw_inputs[idx];
  //     if (name.find ("_lora_") != std::string::npos) {
  //       int size = GraphParser::get_tensor_size (info);
  //       memcpy (std::get<uint16_t *> (generation_inputs[idx]), data_ptr, size);
  //       data_ptr += size;
  //     }
  //   }
  //   munmap (mapped, file_size);
  //   close (fd);

  //   std::cout << "LoRA weights loaded from: " << lora_path << std::endl;
  // }

  this->fresh_kvs.clear ();
  this->kvs.clear ();
  this->kv_sizes.clear ();
  this->kv_row_lengths.clear ();
  this->kv_columns.clear ();

  for (size_t idx = 0; idx < generation_graph_info.raw_inputs.size (); idx++) {
    const auto &[name, info] = generation_graph_info.raw_inputs[idx];
    if (name.find ("past_") == 0) {
      auto *kv_ptr = std::get<uint8_t *> (generation_inputs[idx]);
      int size = GraphParser::get_tensor_size (info);
      size_t kv_input_index = this->kvs.size ();
      this->kvs.push_back ((uint16_t *)kv_ptr);
      this->kv_sizes.push_back (size);

      if (kv_input_index % 2 == 0) {
        this->kv_columns.push_back (info.dimensions[2]);
        this->kv_row_lengths.push_back (info.dimensions.back ());
      }

      // Layer 별 zero point 로 fresh_kv 채움
      int zq = -info.offset;
      if (info.data_type == "QNN_DATATYPE_UFIXED_POINT_8") {
        zq = std::max (0, std::min (255, zq));
        auto *fresh_kv = (uint8_t *)allocate (size); // 또는 너희 allocator
        std::memset (fresh_kv, zq, size); // memset 은 byte-level
        allocated_ptrs_.insert (fresh_kv);
        this->fresh_kvs.push_back ((uint16_t *)fresh_kv);
      } else {
        zq = std::max (0, std::min (65535, zq));
        auto *fresh_kv = (uint16_t *)allocate (size);
        std::fill_n (fresh_kv, size / 2, static_cast<uint16_t> (zq));
        allocated_ptrs_.insert (fresh_kv);
        this->fresh_kvs.push_back (fresh_kv);
      }
      continue; // 아래 fall through 하지 않도록
    }

    if (name.find ("per_layer") == 0) {
      auto *per_layer_embedding_ptr = std::get<uint16_t *> (generation_inputs[idx]);
      int size = GraphParser::get_tensor_size (info);
      this->per_layer_embedding.push_back ((uint16_t *)per_layer_embedding_ptr);
      this->per_layer_embedding_size.push_back (size);
    }
  }


  int prefill_kv_outputs = 0;
  for (const auto &[name, info] : prefill_graph_info.raw_outputs) {
    if (name.find("past_") == 0) {
      prefill_kv_outputs++;
    }
  }
  int generation_kv_outputs = 0;
  for (const auto &[name, info] : generation_graph_info.raw_outputs) {
    if (name.find("past_") == 0) {
      generation_kv_outputs++;
    }
  }
  if ((int)this->kvs.size() != prefill_kv_outputs ||
      (int)this->kvs.size() != generation_kv_outputs) {
    throw std::runtime_error("Gemma4 KV tensor count mismatch: inputs=" +
                             std::to_string(this->kvs.size()) +
                             ", prefill_outputs=" +
                             std::to_string(prefill_kv_outputs) +
                             ", generation_outputs=" +
                             std::to_string(generation_kv_outputs));
  }

  for (size_t idx = 0; idx < prefill_graph_info.raw_inputs.size(); idx++) {
    const auto &[name, info] = prefill_graph_info.raw_inputs[idx];
    if (name.find("past_") != 0) {
      continue;
    }

    int size = GraphParser::get_tensor_size(info);
    if (info.data_type == "QNN_DATATYPE_UFIXED_POINT_8") {
      std::fill_n(std::get<uint8_t *>(prefill_inputs[idx]), size, 128);
    } else {
      std::memset(std::get<uint16_t *>(prefill_inputs[idx]), 0, size);
    }
  }

  const auto &logits_info = GraphParser::get_tensor_info_or_throw (
      generation_graph_info.raw_outputs, "logits");
  logit_scale = logits_info.scale;
  logit_offset = logits_info.offset;

  for (size_t idx = 0; idx < prefill_graph_info.raw_inputs.size (); idx++) {
    const auto &[name, info] = prefill_graph_info.raw_inputs[idx];
    if (name.find ("past_") != 0)
      continue;

    int size = GraphParser::get_tensor_size (info);
    // QNN 컨벤션: f = scale * (q + offset). f=0 ⇒ q = -offset.
    if (info.data_type == "QNN_DATATYPE_UFIXED_POINT_8") {
      int zq = -info.offset;
      zq = std::max (0, std::min (255, zq));
      std::fill_n (std::get<uint8_t *> (prefill_inputs[idx]), size,
          static_cast<uint8_t> (zq));
    } else if (info.data_type == "QNN_DATATYPE_UFIXED_POINT_16") {
      int zq = -info.offset;
      zq = std::max (0, std::min (65535, zq));
      std::fill_n (std::get<uint16_t *> (prefill_inputs[idx]), size / 2,
          static_cast<uint16_t> (zq));
    }
  }

  std::cout << "[ROPE-DBG] cos cache (pos=0, freq 0..7): ";
  for (int i = 0; i < 8; ++i)
    std::cout << position_ids_cos[i] << " ";
  std::cout << "\n[ROPE-DBG] sin cache (pos=0, freq 0..7): ";
  for (int i = 0; i < 8; ++i)
    std::cout << position_ids_sin[i] << " ";
  std::cout << "\n[ROPE-DBG] cos cache (pos=1, freq 0..7): ";
  for (int i = pos_dim; i < pos_dim + 8; ++i)
    std::cout << position_ids_cos[i] << " ";
  std::cout << "\n[ROPE-DBG] sin cache (pos=1, freq 0..7): ";
  for (int i = pos_dim; i < pos_dim + 8; ++i)
    std::cout << position_ids_sin[i] << " ";
  std::cout << "\n";

  LOGD ("----------------------- initialize() done");
}

void
causallm::Gemma4_E2B_QNN::setupParameters (json &cfg, json &generation_cfg, json &nntr_cfg)
{
  // Call base class setupParameters first
  Quick_Dot_AI_QNN::setupParameters(cfg, generation_cfg, nntr_cfg);

  // Read config parameters - model dimensions
  num_hidden_layers = cfg["num_hidden_layers"].get<int>();
  // max_window_layers = cfg["max_window_layers"].get<int>();
  hidden_size = cfg["hidden_size"].get<int>();
  // sequence_length = cfg["sequence_length"].get<int>();
  vocab_size = cfg["vocab_size"].get<int>();
  max_seq_len = cfg["max_seq_len"].get<int>();
  sliding_window = cfg["sliding_window"].get<int>();
  local_rope_theta = cfg["local_rope_theta"].get<float>();
  rope_theta = cfg["rope_theta"].get<float>();
  context_size = cfg["context_size"].get<int>();
  g_head_dim = cfg["global_head_dim"].get<int>();
  l_head_dim = cfg["head_dim"].get<int>();
  head_dim = g_head_dim;

  // Read generation_config parameters
  padding_token = generation_cfg["pad_token_id"].get<int>();
  eos_tokens = generation_cfg["eos_token_id"].get<std::vector<int>>();
  temperature = generation_cfg["temperature"].get<float>();
  top_k = generation_cfg["top_k"].get<int>();
  top_p = generation_cfg["top_p"].get<float>();
  repetition_penalty = generation_cfg.value("repetition_penalty", 1.0f);
  logit_scale = generation_cfg.value("logit_scale", 1.0f);
  logit_offset = generation_cfg.value("logit_offset", 0);

  // Read optional lora_path
  lora_path = nntr_cfg.value("lora_path", "");
  lora_path = rebase_relative_to_model_file(lora_path, model_file_name);
  ple_file_name = nntr_cfg.value("ple_file_name", "");
  ple_file_name = rebase_relative_to_model_file(ple_file_name, model_file_name);
}

void causallm::Gemma4_E2B_QNN::run(const WSTR prompt, bool do_sample,
                                 const WSTR system_prompt,
                                 const WSTR tail_prompt, bool log_output) {
  last_output_.clear();
  stop_requested_.store(false, std::memory_order_release);

  std::string prefill_graph = graphs_to_use[0];
  std::string generation_graph = graphs_to_use[1];

  auto &prefill_inputs = models[prefill_graph].model_inputs;
  auto &generation_inputs = models[generation_graph].model_inputs;

  auto &prefill_model = models[prefill_graph].model_handle;
  auto &generation_model = models[generation_graph].model_handle;

  std::cout << "this->kv.size() : "<<this->kvs.size()<<std::endl;
  for (int i = 0; i < this->kvs.size(); i++) {
    std::memcpy(this->kvs[i], this->fresh_kvs[i], this->kv_sizes[i]);
  }

  auto _input = tokenizer->Encode(prompt);
  auto token  = _input.back();
  std::cout << prompt << std::endl;

  
  // --- DEBUG ---
  std::cout << "[TOK-DBG] prompt len=" << _input.size () << " first 16 tokens: ";
  for (int i = 0; i < std::min<int> (_input.size (), 16); ++i)
    std::cout << _input[i] << " ";
  std::cout << "\n";
  std::cout << "[TOK-DBG] last 8 tokens: ";
  for (int i = std::max (0, (int)_input.size () - 8); i < (int)_input.size (); ++i)
    std::cout << _input[i] << " ";
  std::cout << "\n";

  // 디코드해서 확인
  std::cout << "[TOK-DBG] decoded first 16: '"
            << tokenizer->Decode (std::vector<int> (_input.begin (),
                   _input.begin () + std::min<size_t> (_input.size (), 16)))
            << "'\n";
  // -------------

  unsigned int _len = _input.size() - 1;
  if(_len <= 0){
    std::cout << "[Error] Input is empty or invalid" << std::endl;
    return;
  }

  auto _n_chunks = (_len % 256 != 0) ? ((_len / 256) + 1) : (_len / 256);
  std::cout << "n_chunk: " << _n_chunks << ", len: " << _len << std::endl;


  std::vector<int> output;
  std::vector<ml::train::TensorDim::IO_TensorType> outputs;

  for(int c = 0; c < _n_chunks; c++) {
    int _chunk_len = ((c + 1) * 256 < _len) ? context_size : (_len - (c * 256));
    int kv_len = c * context_size;

    for(int i = 0; i < context_size; i++)
      input_sample[i] = (i < _chunk_len) ? _input[c * 256 + i] : padding_token;

    fill_attention_mask_with_length(context_size, prefill_attention_mask_columns,
                                    _chunk_len, attention_mask);
    fill_attention_mask_with_prev_length(context_size,
                                         prefill_attention_mask_columns,
                                         std::min(kv_len,
                                                  generation_full_kv_past_length),
                                         attention_mask);

    fill_attention_mask_with_length(context_size,
                                    prefill_sliding_attention_mask_columns,
                                    _chunk_len, sliding_attention_mask);
    fill_attention_mask_with_prev_length(
        context_size, prefill_sliding_attention_mask_columns,
        std::min(kv_len, generation_sliding_kv_past_length),
        sliding_attention_mask);

    std::fill_n(prefill_position_ids_cos, context_size * pos_dim, 65535);
    std::fill_n(prefill_position_ids_sin, context_size * pos_dim, 32768);
    std::fill_n(prefill_swa_position_ids_cos, context_size * swa_pos_dim, 65535);
    std::fill_n(prefill_swa_position_ids_sin, context_size * swa_pos_dim, 32768);

    auto pos_ids_offset = c * context_size * pos_dim;
    auto swa_pos_ids_offset = c * context_size * swa_pos_dim;
    std::memcpy(prefill_position_ids_cos, position_ids_cos + pos_ids_offset, _chunk_len * pos_dim * sizeof(uint16_t));
    std::memcpy(prefill_position_ids_sin, position_ids_sin + pos_ids_offset, _chunk_len * pos_dim * sizeof(uint16_t));
    std::memcpy(prefill_swa_position_ids_cos,
                swa_position_ids_cos + swa_pos_ids_offset,
                _chunk_len * swa_pos_dim * sizeof(uint16_t));
    std::memcpy(prefill_swa_position_ids_sin,
                swa_position_ids_sin + swa_pos_ids_offset,
                _chunk_len * swa_pos_dim * sizeof(uint16_t));


    fill_prefill_ple_chunk_ (_input, c, _chunk_len);
    std::cout << "[MASK-DBG] attention_mask first 8 (valid?): ";
    for (int i = 0; i < 8; ++i)
      std::cout << attention_mask[i] << " ";
    std::cout << "\n[MASK-DBG] attention_mask at " << _chunk_len << ".."
              << _chunk_len + 8 << " (should be masked): ";
    for (int i = _chunk_len; i < _chunk_len + 8; ++i)
      std::cout << attention_mask[i] << " ";
    std::cout << "\n";

    outputs = prefill_model->inference (1, prefill_inputs);

#pragma omp parallel for
    for (int i = 0; i < (int)this->kv_row_lengths.size () * 2; i++) {
      bool is_value = i % 2 == 0;
      bool is_key = !is_value;
      int layer_idx = i / 2;
      int kv_idx = layer_idx * 2 + (is_key ? 0 : 1);
      int dest_row_length = kv_row_lengths[layer_idx];
      bool is_sliding = dest_row_length == generation_sliding_kv_past_length;
      int src_row_length = context_size;

      auto output = std::get<uint8_t *>(outputs[i]);
      auto dest = (uint8_t *)this->kvs[kv_idx];
      int num_column = kv_columns[layer_idx];

      int target_idx = kv_len;
      if (is_sliding && kv_len + _chunk_len > dest_row_length) {
        target_idx = dest_row_length - _chunk_len;
        if (is_key) {
          for (int col = 0; col < num_column; ++col) {
            uint8_t *col_base = dest + col * dest_row_length;
            std::memmove(col_base, col_base + _chunk_len,
                         dest_row_length - _chunk_len);
          }
        } else {
          std::memmove(dest, dest + _chunk_len * num_column,
                       (dest_row_length - _chunk_len) * num_column);
        }
      }

      if (is_key) {
        process_key(output, _chunk_len, num_column, dest, target_idx,
                    dest_row_length, src_row_length);
      } else {
        process_value(output, _chunk_len, num_column, dest, target_idx);
      }
    };
  }

  std::fill_n(generation_attention_mask, generation_attention_mask_elements, 0);
  std::fill_n(generation_sliding_attention_mask,
              generation_sliding_attention_mask_elements, 0);

  generation_attention_mask[generation_attention_mask_elements - 1] =
      std::numeric_limits<uint16_t>::max();
  generation_sliding_attention_mask[generation_sliding_attention_mask_elements -
                                    1] = std::numeric_limits<uint16_t>::max();

  for (int i = 0; i < _len && i < generation_full_kv_past_length; i++)
    generation_attention_mask[i] = std::numeric_limits<uint16_t>::max();
  for (int i = 0; i < _len && i < generation_sliding_kv_past_length; i++)
    generation_sliding_attention_mask[i] = std::numeric_limits<uint16_t>::max();

  auto start = std::chrono::system_clock::now();
  int idx;
  int prefill_len = _len;
  for (idx = prefill_len; idx < generation_full_kv_past_length; idx++) {
    generation_sample[0] = token;

    generation_attention_mask[idx] = std::numeric_limits<uint16_t>::max();
    if (idx < generation_sliding_kv_past_length) {
      generation_sliding_attention_mask[idx] =
          std::numeric_limits<uint16_t>::max();
    }
    std::memcpy(generation_position_ids_cos, position_ids_cos + idx * pos_dim,
                pos_dim * sizeof(uint16_t));
    std::memcpy(generation_position_ids_sin, position_ids_sin + idx * pos_dim,
                pos_dim * sizeof(uint16_t));
    std::memcpy(generation_swa_position_ids_cos,
                swa_position_ids_cos + idx * swa_pos_dim,
                swa_pos_dim * sizeof(uint16_t));
    std::memcpy(generation_swa_position_ids_sin,
                swa_position_ids_sin + idx * swa_pos_dim,
                swa_pos_dim * sizeof(uint16_t));

    fill_generation_ple_(token);
    
    outputs = generation_model->inference (1, generation_inputs);    

    if(idx > prefill_len) {
#pragma omp parallel for
      for (int i = 0; i < (int)this->kv_row_lengths.size() * 2; i++) {
        bool is_value = i % 2 == 0;
        bool is_key = !is_value;
        int layer_idx = i / 2;
        int kv_idx = layer_idx * 2 + (is_key ? 0 : 1);
        int dest_row_length = kv_row_lengths[layer_idx];
        bool is_sliding = dest_row_length == generation_sliding_kv_past_length;

        auto output = std::get<uint8_t *>(outputs[i]);
        auto dest = (uint8_t *)this->kvs[kv_idx];
        int num_column = kv_columns[layer_idx];
        int target_idx = idx;

        if (is_sliding && (idx + 1) > dest_row_length) {
          target_idx = dest_row_length - 1;
          if (is_key) {
            for (int col = 0; col < num_column; ++col) {
              uint8_t *col_base = dest + col * dest_row_length;
              std::memmove(col_base, col_base + 1, dest_row_length - 1);
            }
          } else {
            std::memmove(dest, dest + num_column,
                         (dest_row_length - 1) * num_column);
          }
        }

        if (is_key) {
          process_key(output, 1, num_column, dest, target_idx, dest_row_length,
                      1);
        } else {
          process_value(output, 1, num_column, dest, target_idx);
        }
      }
    }
    
    token = sample (std::get<uint16_t *> (outputs.back ()), vocab_size,
        _input.data (), _input.size (), logit_scale, logit_offset,
        repetition_penalty, temperature, top_p, top_k);

    output.push_back(token);

    bool reached_eos = false;
    for(auto eos : eos_tokens){
      if (token == eos) {
        reached_eos = true;
        break;
      }
    }
    if (reached_eos) {
      std::cout << "Finished generating, break..." << std::endl;
      break;
    }

    std::string decoded = tokenizer->Decode ({ token });
    last_output_ += decoded;
    LOGD ("%d : %s", token, decoded.c_str ());
    if (streamer_) {
      if (streamer_put (streamer_, decoded.c_str ()) != 0) {
        stop_requested_.store(true, std::memory_order_release);
        break;
      }
    } else if (log_output) {
      std::cout << decoded << std::flush;
    }
    _input.push_back (token);

    if (stop_requested_.load(std::memory_order_acquire)) {
      break;
    }
  }

  if (streamer_) {
    streamer_end(streamer_);
  }

  has_run_ = true;
  
  auto end = std::chrono::system_clock::now();
  raw_exec_seconds = end - start;
  if (log_output) {  
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "Generation exec_time : " << raw_exec_seconds.count()
            << ", token per second: " << (idx - _len) / raw_exec_seconds.count()
            << ", token generation time average: "
            << raw_exec_seconds.count() / (idx - _len) << std::endl;
  }
}

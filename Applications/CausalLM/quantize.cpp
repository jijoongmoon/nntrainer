// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file   quantize.cpp
 * @date   04 March 2026
 * @brief  Quantization utility for CausalLM models.
 *         Reads a FP32 model and converts weights to a target data type,
 *         saving the quantized weights (.bin or .safetensors) and a new
 *         nntr_config.json.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @usage
 *   nntr_quantize <model_path> [options]
 *
 *   Required:
 *     <model_path>        Path to the model directory containing:
 *                           config.json, generation_config.json,
 *                           nntr_config.json, and the .bin weight file.
 *
 *   Options:
 *     --output, -o <path> Output directory (default: <model_path>)
 *     --fc_dtype <type>   Target dtype for FC layers (default: Q4_0)
 *     --embd_dtype <type> Target dtype for embedding layer (default: FP32)
 *     --lmhead_dtype <type> Target dtype for LM head layer (default: FP32)
 *     --output_bin <name> Output weight filename (auto-generated if omitted)
 *     --output_format <fmt> Output container: 'bin' (default) or 'safetensors'
 *     --sidecar <mode>    mmap sidecar packaging of the embedding-0 and
 *                         per-layer-embedding lookup tables:
 *                         auto (default) | on | off.
 *                         Only an UNTIED table saved as Q4_0 / Q6_K can be
 *                         split, so with the default --embd_dtype FP32 'auto'
 *                         splits nothing and emits the single-file package.
 *
 *   Supported data types: FP32, FP16, Q4_0, Q4_K, Q6_K
 *
 *   Example:
 *     # Quantize Qwen3-4B to Q4_0 FC layers (embedding stays FP32):
 *     nntr_quantize /path/to/qwen3-4b --fc_dtype Q4_0
 *
 *     # Quantize with Q6_K embedding and Q4_0 FC layers:
 *     nntr_quantize /path/to/qwen3-4b --fc_dtype Q4_0 --embd_dtype Q6_K
 *
 *     # Quantize to a different output directory:
 *     nntr_quantize /path/to/qwen3-4b -o /output/qwen3-4b-q4
 *
 *     # Use a target nntr_config.json directly:
 *     nntr_quantize /path/to/qwen3-4b --config /path/to/target_nntr_config.json
 *
 *     # Require the mmap sidecar package (untied lookup table + Q4_0/Q6_K):
 *     nntr_quantize /path/to/qwen3-4b --embd_dtype Q6_K --sidecar on
 */

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <factory.h>
#include <nntrainer_error.h>
#include <tensor_dim.h>

#include "causal_lm.h"
#include "deberta_v2.h"
#include "embedding_gemma.h"
#include "gemma2_causallm.h"
#include "gemma3_causallm.h"
#include "gemma4_causallm.h"
#if !defined(_WIN32)
#include "gptoss_cached_slim_causallm.h"
#endif
#include "gptoss_causallm.h"
#include "lfm2_causallm.h"
#if !defined(_WIN32) && !defined(__ANDROID__)
#include "multilingual_tinybert_16mb.h"
#endif
#include "qwen2_causallm.h"
#include "qwen2_embedding.h"
#include "xlm_roberta/xlm_roberta.h"
#if !defined(_WIN32)
#include "qwen3_cached_slim_moe_causallm.h"
#endif
#include "qwen3_causallm.h"
#include "qwen3_embedding.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"

using json = nlohmann::json;
using DataType = ml::train::TensorDim::DataType;

namespace {

/**
 * @brief Map of string data type names to DataType enum values
 */
const std::map<std::string, DataType> dtype_str_map = {
  {"FP32", DataType::FP32}, {"FP16", DataType::FP16},
  {"Q4_0", DataType::Q4_0}, {"Q6_K", DataType::Q6_K},
  {"Q4_K", DataType::Q4_K}, {"QS4CX", DataType::QS4CX},
  {"NONE", DataType::NONE}};

/**
 * @brief Map of string ISA names to ISA enum values
 */
const std::map<std::string, ml::train::ISA> isa_str_map = {
  {"DEFAULT", ml::train::ISA::DEFAULT},
  {"X86", ml::train::ISA::X86},
  {"ARM", ml::train::ISA::ARM},
};

/**
 * @brief Convert string to ISA enum
 */
ml::train::ISA strToISA(const std::string &s) {
  std::string upper = s;
  std::transform(upper.begin(), upper.end(), upper.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  auto it = isa_str_map.find(upper);
  if (it == isa_str_map.end()) {
    throw std::invalid_argument("Unsupported ISA: " + s +
                                ". Supported: DEFAULT, X86, ARM");
  }
  return it->second;
}

/**
 * @brief Convert ISA enum to string
 */
std::string isaToStr(ml::train::ISA isa) {
  for (const auto &[key, val] : isa_str_map) {
    if (val == isa)
      return key;
  }
  return "DEFAULT";
}

/**
 * @brief Convert string to DataType enum
 */
DataType strToDataType(const std::string &s) {
  std::string upper = s;
  std::transform(upper.begin(), upper.end(), upper.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  auto it = dtype_str_map.find(upper);
  if (it == dtype_str_map.end()) {
    throw std::invalid_argument("Unsupported data type: " + s +
                                ". Supported: FP32, FP16, Q4_0, Q6_K, Q4_K");
  }
  return it->second;
}

/**
 * @brief Convert DataType enum to string
 */
std::string dataTypeToStr(DataType dt) {
  for (const auto &[key, val] : dtype_str_map) {
    if (val == dt)
      return key;
  }
  return "NONE";
}

/**
 * @brief Build model_tensor_type string from fc_dtype and activation dtype
 *        Format: "<weight_type>-<activation_type>"
 */
std::string buildModelTensorType(const std::string &fc_dtype) {
  return fc_dtype + "-FP32";
}

/**
 * @brief Return lowercase copy of a string
 */
std::string toLower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

/**
 * @brief Generate a descriptive output bin filename
 */
std::string generateOutputBinName(const std::string &original_bin,
                                  const std::string &fc_dtype,
                                  const std::string &embd_dtype,
                                  const std::string &target_isa) {
  // Extract model name from original (e.g., "nntr_qwen3_4b_fp32.bin" ->
  // "nntr_qwen3_4b")
  std::string base = original_bin;
  // Remove .bin extension
  auto dot_pos = base.rfind(".bin");
  if (dot_pos != std::string::npos)
    base = base.substr(0, dot_pos);

  // Remove old dtype suffix patterns (e.g., _fp32, _q40_fp32)
  // Common patterns: _fp32, _fp16, _q40, _q6k, _q4k, etc.
  std::vector<std::string> dtype_suffixes = {"_fp32", "_fp16",  "_q40",
                                             "_q4_0", "_q6k",   "_q6_k",
                                             "_q4k",  "_qs4cx", "_q4_k"};
  for (const auto &suffix : dtype_suffixes) {
    auto pos = base.rfind(suffix);
    if (pos != std::string::npos && pos + suffix.size() == base.size()) {
      base = base.substr(0, pos);
      break;
    }
  }

  // Build new dtype suffix
  std::string fc_lower = fc_dtype;
  std::transform(fc_lower.begin(), fc_lower.end(), fc_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  // Replace _ for cleaner naming
  std::string fc_clean = fc_lower;
  fc_clean.erase(std::remove(fc_clean.begin(), fc_clean.end(), '_'),
                 fc_clean.end());

  std::string embd_lower = embd_dtype;
  std::transform(embd_lower.begin(), embd_lower.end(), embd_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::string embd_clean = embd_lower;
  embd_clean.erase(std::remove(embd_clean.begin(), embd_clean.end(), '_'),
                   embd_clean.end());

  if (embd_clean == fc_clean) {
    return base + "_" + fc_clean + "_" + target_isa + ".bin";
  }
  return base + "_" + fc_clean + "_embd" + embd_clean + "_" + target_isa +
         ".bin";
}

// ===========================================================================
// mmap sidecar packaging
//
// INVARIANT (stated as narrowly as the code actually guarantees it): when a
// lookup table is ELIGIBLE, a generated package carries it as an mmap'd
// sidecar unless explicitly asked not to, and the union of the emitted files
// is byte-for-byte the single-file package. Eligible means all of:
//
//   * the table is one of the two the runtime has a config key for
//     (embedding0 -> embedding_file_name, per_layer_input_embedding ->
//     ple_file_name); and
//   * it is an UNTIED "embedding_layer" graph node, not "tie_word_embeddings";
//     and
//   * it is saved in a row-block dtype the manifest loader reads --
//     Q4_0 (out_dim % 32 == 0) or Q6_K (out_dim % 256 == 0), i.e.
//     --embd_dtype Q4_0 / Q6_K; and
//   * the output container is .bin.
//
// This is NARROWER than "the default package is now split": --embd_dtype
// defaults to FP32, and FP32 has no row-block encoding, so a run that does not
// ask for a quantized embedding still emits the single-file package under
// `auto`. What flipped is the DEFAULT DECISION for eligible tables, not the
// shape of every package.
//
// The runtime half has been in the tree for a while (embedding_layer.cpp reads
// the manifests, transformer.cpp / gemma4_causallm.cpp consume the
// embedding_file_name / ple_file_name keys); the missing half was that nothing
// EMITTED a package shaped that way, so the default package never got the
// smaller resident plane. This is that half.
//
// Only the two tables the runtime has config keys for are candidates. Other
// lookup tables in the tree (position_embedding, token_type_embedding) have no
// key to point a manifest at, so splitting them out would produce a package
// nothing can load.
// ===========================================================================

/**
 * @brief Sidecar packaging mode
 */
enum class SidecarMode {
  OFF,  /**< never split: emit the single-file package */
  AUTO, /**< default: split every table that can be split, report the rest */
  ON    /**< require the split: fail if any candidate table cannot be split */
};

/**
 * @brief One sidecar candidate: a lookup table and the config key that points
 *        the runtime at its manifest
 */
struct SidecarCandidate {
  const char *layer_name; /**< graph node name */
  const char *config_key; /**< nntr_config.json key holding the manifest path */
  const char *suffix;     /**< manifest/payload filename suffix */
};

const std::vector<SidecarCandidate> sidecar_candidates = {
  {"embedding0", "embedding_file_name", "_embd"},
  {"per_layer_input_embedding", "ple_file_name", "_ple"},
};

/**
 * @brief Convert a sidecar mode string to the enum
 */
SidecarMode strToSidecarMode(const std::string &s) {
  const std::string lower = toLower(s);
  if (lower == "off" || lower == "false" || lower == "0" || lower == "none")
    return SidecarMode::OFF;
  if (lower == "auto")
    return SidecarMode::AUTO;
  if (lower == "on" || lower == "true" || lower == "1" || lower == "require")
    return SidecarMode::ON;
  throw std::invalid_argument("Unsupported sidecar mode: " + s +
                              ". Supported: auto, on, off");
}

/**
 * @brief Bytes one quantized row occupies, for the row-block dtypes the
 *        sidecar manifest loader accepts
 * @return 0 when @a dt has no sidecar row-block encoding
 * @note   Mirrors loadGgmlManifest() in layers/embedding_layer.cpp. The block
 *         divisibility is a hard precondition there (it throws), so it is
 *         checked here rather than shipped as a package that throws on load.
 */
size_t ggmlRowBytes(DataType dt, unsigned int out_dim) {
  if (dt == DataType::Q4_0)
    return (out_dim % 32 == 0) ? static_cast<size_t>(out_dim) / 32 * 18 : 0;
  if (dt == DataType::Q6_K)
    return (out_dim % 256 == 0) ? static_cast<size_t>(out_dim) / 256 * 210 : 0;
  return 0;
}

/**
 * @brief The manifest "datatype" string for a row-block dtype
 */
std::string sidecarDatatypeName(DataType dt) {
  return (dt == DataType::Q6_K) ? "q6_k" : "q4_0";
}

/**
 * @brief A planned sidecar emission
 */
struct SidecarPlan {
  std::string layer_name;    /**< graph node name to route */
  std::string config_key;    /**< key to write into nntr_config.json */
  std::string manifest_name; /**< manifest filename (relative) */
  std::string payload_name;  /**< payload filename (relative) */
  DataType dtype;            /**< dtype the writer will use for this layer */
  unsigned int rows;         /**< table row count (in_dim) */
  unsigned int size;         /**< elements per row (out_dim) */
  size_t row_bytes;          /**< payload bytes per row */
};

/**
 * @brief Strip the container extension from an output weight filename
 */
std::string stripWeightExtension(const std::string &name) {
  for (const std::string &ext :
       {std::string(".safetensors"), std::string(".bin")}) {
    if (name.size() > ext.size() &&
        name.compare(name.size() - ext.size(), ext.size(), ext) == 0)
      return name.substr(0, name.size() - ext.size());
  }
  return name;
}

/**
 * @brief Resolve architecture name from config
 */
std::string resolve_architecture(std::string model_type,
                                 const std::string &architecture) {
  std::transform(model_type.begin(), model_type.end(), model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (model_type == "embedding") {
    // Already-resolved nntrainer class names — pass through
    if (architecture == "Qwen3Embedding" || architecture == "Qwen2Embedding" ||
        architecture == "EmbeddingGemma" ||
        architecture == "MultilingualTinyBert" || architecture == "DebertaV2" ||
        architecture == "XLMRobertaForMaskedLM")
      return architecture;

    if (architecture == "Qwen3ForCausalLM")
      return "Qwen3Embedding";
    else if (architecture == "Gemma3ForCausalLM" ||
             architecture == "Gemma3TextModel")
      return "EmbeddingGemma";
    else if (architecture == "Qwen2Model")
      return "Qwen2Embedding";
    else if (architecture == "BertForMaskedLM")
      return "MultilingualTinyBert";
    else if (architecture == "XLMRobertaModel")
      return "XLMRobertaForMaskedLM";
    else if (architecture == "DebertaV2ForMaskedLM")
      return "DebertaV2";
    else
      throw std::invalid_argument(
        "Unsupported architecture for embedding model: " + architecture);
  }

  if (architecture == "Gemma4ForConditionalGeneration") {
    return "Gemma4ForCausalLM";
  }

  return architecture;
}

/**
 * @brief Return the final component of a dotted Python class name
 */
std::string getLastComponent(const std::string &type) {
  const size_t last_dot_pos = type.find_last_of('.');
  if (last_dot_pos == std::string::npos)
    return type;

  return type.substr(last_dot_pos + 1);
}

/**
 * @brief Register all CausalLM model factories
 */
void registerAllModels() {
  auto &factory = causallm::Factory::Instance();

  factory.registerModel("LlamaForCausalLM", [](json cfg, json generation_cfg,
                                               json nntr_cfg) {
    return std::make_unique<causallm::CausalLM>(cfg, generation_cfg, nntr_cfg);
  });
  factory.registerModel("Qwen2ForCausalLM",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::Qwen2CausalLM>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel("Qwen2Embedding",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::Qwen2Embedding>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel("Qwen3ForCausalLM",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::Qwen3CausalLM>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel("Qwen3MoeForCausalLM",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::Qwen3MoECausalLM>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel("Qwen3SlimMoeForCausalLM", [](json cfg,
                                                      json generation_cfg,
                                                      json nntr_cfg) {
    return std::make_unique<causallm::Qwen3SlimMoECausalLM>(cfg, generation_cfg,
                                                            nntr_cfg);
  });
#if !defined(_WIN32)
  factory.registerModel(
    "Qwen3CachedSlimMoeForCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3CachedSlimMoECausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
#endif
  factory.registerModel("Qwen3Embedding",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::Qwen3Embedding>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel("GptOssForCausalLM",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::GptOssForCausalLM>(
                            cfg, generation_cfg, nntr_cfg);
                        });
#if !defined(_WIN32)
  factory.registerModel(
    "GptOssCachedSlimCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::GptOssCachedSlimCausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
#endif
  factory.registerModel("Gemma2ForCausalLM",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::Gemma2CausalLM>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel("Gemma3ForCausalLM",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::Gemma3CausalLM>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel("Gemma4ForCausalLM",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::Gemma4CausalLM>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel("EmbeddingGemma",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::EmbeddingGemma>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel("Lfm2ForCausalLM",
                        [](json cfg, json generation_cfg, json nntr_cfg) {
                          return std::make_unique<causallm::Lfm2CausalLM>(
                            cfg, generation_cfg, nntr_cfg);
                        });
  factory.registerModel("DebertaV2", [](json cfg, json generation_cfg,
                                        json nntr_cfg) {
    return std::make_unique<causallm::DebertaV2>(cfg, generation_cfg, nntr_cfg);
  });
#if !defined(_WIN32) && !defined(__ANDROID__)
  factory.registerModel(
    "MultilingualTinyBert", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::MultilingualTinyBert>(
        cfg, generation_cfg, nntr_cfg);
    });
#endif
#if !defined(_WIN32)
  factory.registerModel(
    "XLMRobertaForMaskedLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::XLMRobertaForMaskedLM>(
        cfg, generation_cfg, nntr_cfg);
    });
#endif
}

/**
 * @brief Print usage information
 */
void printUsage(const char *prog) {
  std::cout
    << "Usage: " << prog << " <model_path> [options]\n"
    << "\n"
    << "Quantize a CausalLM model from FP32 to a target data type.\n"
    << "\n"
    << "Required:\n"
    << "  <model_path>          Path to model directory containing:\n"
    << "                          config.json, generation_config.json,\n"
    << "                          nntr_config.json, and .bin weight file\n"
    << "\n"
    << "Options:\n"
    << "  --output, -o <path>   Output directory (default: <model_path>)\n"
    << "  --fc_dtype <type>     Target dtype for FC layers (default: Q4_0)\n"
    << "  --embd_dtype <type>   Target dtype for embedding (default: FP32)\n"
    << "  --lmhead_dtype <type> Target dtype for LM head (default: same as "
       "embd_dtype)\n"
    << "  --isa <arch>          Target instruction set architecture for "
       "quantized weights\n"
    << "                        (default: DEFAULT). Options: DEFAULT, X86, "
       "ARM.\n"
    << "  --output_bin <name>   Output weight filename (auto-generated if "
       "omitted)\n"
    << "  --output_format <fmt> Output container: 'bin' (default) or "
       "'safetensors'\n"
    << "  --config <path>       Use a target nntr_config.json instead of\n"
    << "                        individual dtype options. The fc_layer_dtype,\n"
    << "                        embedding_dtype, lmhead_dtype and sidecar\n"
    << "                        fields from this config will be used.\n"
    << "  --sidecar <mode>      mmap sidecar packaging of the embedding-0 and\n"
    << "                        per-layer-embedding lookup tables:\n"
    << "                          auto  split every table that can be split,\n"
    << "                                report the ones that cannot (default)\n"
    << "                          on    require the split; fail if a "
       "candidate\n"
    << "                                table cannot be split\n"
    << "                          off   emit the single-file package\n"
    << "                        A table can be split only when it is UNTIED\n"
    << "                        (an 'embedding_layer' node, not\n"
    << "                        'tie_word_embeddings') AND saved as Q4_0 or\n"
    << "                        Q6_K. --embd_dtype defaults to FP32, which has\n"
    << "                        no sidecar row encoding, so 'auto' splits\n"
    << "                        nothing unless --embd_dtype Q4_0 / Q6_K is\n"
    << "                        also given. 'auto' prints the reason for every\n"
    << "                        table it leaves in the bin.\n"
    << "  --no-sidecar          Alias for --sidecar off\n"
    << "  --help, -h            Show this help message\n"
    << "\n"
    << "Supported data types: FP32, FP16, Q4_0, Q6_K, Q4_K\n"
    << "Supported ISA options: DEFAULT (current platform), X86, ARM\n"
    << "\n"
    << "Examples:\n"
    << "  # Quantize FC layers to Q4_0 (default):\n"
    << "  " << prog << " /path/to/qwen3-4b\n"
    << "\n"
    << "  # Quantize FC layers to Q4_0 and embedding to Q6_K:\n"
    << "  " << prog << " /path/to/qwen3-4b --fc_dtype Q4_0 --embd_dtype Q6_K\n"
    << "\n"
    << "  # Quantize to ARM format for deployment on ARM devices:\n"
    << "  " << prog << " /path/to/qwen3-4b --isa ARM\n"
    << "\n"
    << "  # Quantize to X86 format for deployment on x86 devices:\n"
    << "  " << prog << " /path/to/qwen3-4b --isa X86\n"
    << "\n"
    << "  # Quantize to a different output directory:\n"
    << "  " << prog << " /path/to/qwen3-4b -o /output/qwen3-4b-q4\n"
    << "\n"
    << "  # Use a target nntr_config.json:\n"
    << "  " << prog
    << " /path/to/qwen3-4b --config /path/to/target_nntr_config.json\n"
    << "\n"
    << "  # Require the mmap sidecar package (untied table + Q4_0/Q6_K):\n"
    << "  " << prog << " /path/to/qwen3-4b --embd_dtype Q6_K --sidecar on\n"
    << "\n"
    << "  # Emit the single-file package (no mmap sidecars):\n"
    << "  " << prog << " /path/to/qwen3-4b --no-sidecar\n";
}

/**
 * @brief Build the layer_dtype_map for the model based on target dtypes.
 *
 * Layer naming convention in Transformer:
 *   - embedding0          : embedding layer
 *   - layer{i}_wq/wk/wv  : attention Q/K/V projections (FC layers)
 *   - layer{i}_attention_out : attention output projection (FC layer)
 *   - layer{i}_ffn_up/gate/down : FFN layers (FC layers)
 *   - layer{i}_attention_norm, layer{i}_ffn_norm : RMSNorm layers
 *   - output_norm          : final RMSNorm
 *   - output_of_causallm   : LM head (FC layer)
 *
 * The dtype map assigns:
 *   - embedding0             -> embd_dtype
 *   - All FC layers (wq, wk, wv, attention_out, ffn_*) -> fc_dtype
 *   - output_of_causallm     -> lmhead_dtype
 *   - RMSNorm / other layers -> FP32 (not quantized)
 */
std::map<std::string, DataType>
buildLayerDtypeMap(int num_layers, DataType fc_dtype, DataType embd_dtype,
                   DataType lmhead_dtype, bool include_lmhead) {

  std::map<std::string, DataType> dtype_map;

  // Embedding layer
  if (embd_dtype != DataType::FP32 && embd_dtype != DataType::NONE) {
    dtype_map["embedding0"] = embd_dtype;
    dtype_map["position_embedding"] = embd_dtype;
    dtype_map["token_type_embedding"] = embd_dtype;
  }

  // Gemma4-style PLE input embedding is a lookup table (EmbeddingLayer), whose
  // save supports only Q4_0/Q6_K/FP32 -- NOT the FC matmul packings (QINT4,
  // QS4CX). Key it off embd_dtype so a QINT4/QS4CX fc_dtype run does not choke
  // on this embedding, and so it agrees with the layer's own weight_dtype
  // (gemma4_causallm.cpp requests EMBEDDING_DTYPE).
  if (embd_dtype != DataType::FP32 && embd_dtype != DataType::NONE) {
    dtype_map["per_layer_input_embedding"] = embd_dtype;
  }
  // Gemma4 PLE projection is a plain FC layer -> fc_dtype.
  dtype_map["per_layer_input_projection"] = fc_dtype;

  // Transformer decoder layers
  for (int i = 0; i < num_layers; ++i) {
    std::string prefix = "layer" + std::to_string(i);

    // Attention FC layers
    if (fc_dtype != DataType::FP32 && fc_dtype != DataType::NONE) {
      dtype_map[prefix + "_wq"] = fc_dtype;
      dtype_map[prefix + "_wk"] = fc_dtype;
      dtype_map[prefix + "_wv"] = fc_dtype;
      dtype_map[prefix + "_attention_out"] = fc_dtype;

      // Attention Gates
      dtype_map[prefix + "_attention_gate_down"] = fc_dtype;
      dtype_map[prefix + "_attention_gate_up"] = fc_dtype;

      // Attention Gates
      dtype_map[prefix + "_attention_gate_down"] = fc_dtype;
      dtype_map[prefix + "_attention_gate_up"] = fc_dtype;

      // FFN FC layers - version4
      dtype_map[prefix + "_ffn_gate_up"] = fc_dtype;
      dtype_map[prefix + "_ffn_gate_down"] = fc_dtype;
      dtype_map[prefix + "_ffn_linear_up"] = fc_dtype;

      // FFN FC layers - version3 (Qwen/Gemma LLMs)
      dtype_map[prefix + "_ffn_gate"] = fc_dtype;
      dtype_map[prefix + "_ffn_up"] = fc_dtype;
      dtype_map[prefix + "_ffn_down"] = fc_dtype;

      dtype_map[prefix + "_ffn_output"] = fc_dtype;

      // LFM2 conv-block projections (causal_conv1d core stays FP32, but the
      // in/out projections follow fc_layer_dtype like any other FC layer).
      dtype_map[prefix + "_conv_in_proj"] = fc_dtype;
      dtype_map[prefix + "_conv_out_proj"] = fc_dtype;

      // FFN FC layers - BERT (BertTransformer)
      dtype_map[prefix + "_ffn_fc1"] = fc_dtype;

      // FFN FC layers - DeBERTa V2
      dtype_map[prefix + "_intermediate"] = fc_dtype;
      dtype_map[prefix + "_output_dense"] = fc_dtype;

      // for PLE
      dtype_map[prefix + "_per_layer_input_gate"] = fc_dtype;
      dtype_map[prefix + "_per_layer_input_proj"] = fc_dtype;
      // Per-layer PLE table: a lookup table like per_layer_input_embedding
      // above, so embd_dtype (NOT an FC matmul packing).
      if (embd_dtype != DataType::FP32 && embd_dtype != DataType::NONE) {
        dtype_map[prefix + "_ple"] = embd_dtype;
      }

      dtype_map[prefix + "_ple_projection"] = fc_dtype;
      dtype_map[prefix + "_ple_input_gate"] = fc_dtype;
    }
  }

  // LM Head layer
  if (include_lmhead && lmhead_dtype != DataType::FP32 &&
      lmhead_dtype != DataType::NONE) {
    dtype_map["output_of_causallm"] = lmhead_dtype;
  }

  return dtype_map;
}

/**
 * @brief Add SentenceTransformer module dtype overrides to the dtype map
 */
void addSentenceTransformerLayerDtypes(std::map<std::string, DataType> &map,
                                       const json &nntr_cfg,
                                       const std::string &model_path,
                                       DataType fc_dtype) {
  if (fc_dtype == DataType::FP32 || fc_dtype == DataType::NONE ||
      !nntr_cfg.contains("module_config_path")) {
    return;
  }

  std::filesystem::path modules_config_path =
    nntr_cfg["module_config_path"].get<std::string>();
  if (modules_config_path.is_relative()) {
    modules_config_path =
      std::filesystem::path(model_path) / modules_config_path;
  }

  json modules_json = causallm::LoadJsonFile(modules_config_path.string());
  auto modules = modules_json.get<std::vector<json>>();
  for (const auto &module : modules) {
    if (!module.contains("type"))
      continue;

    const std::string component =
      getLastComponent(module["type"].get<std::string>());
    if (component != "Dense")
      continue;

    std::string layer_name;
    if (module.contains("name")) {
      layer_name = module["name"].get<std::string>();
    } else if (module.contains("idx")) {
      layer_name = "sentence_module_" +
                   std::to_string(module["idx"].get<int>()) + "_" + component;
    } else {
      throw std::runtime_error(
        "Dense SentenceTransformer module has neither name nor idx.");
    }

    map[layer_name] = fc_dtype;
  }
}

} // anonymous namespace

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printUsage(argv[0]);
    return EXIT_FAILURE;
  }

  std::string first_arg = argv[1];
  if (first_arg == "--help" || first_arg == "-h") {
    printUsage(argv[0]);
    return EXIT_SUCCESS;
  }

  // Parse arguments
  std::string model_path = argv[1];
  std::string output_dir = "";
  std::string fc_dtype_str = "Q4_0";
  std::string embd_dtype_str = "FP32";
  std::string lmhead_dtype_str = "";
  std::string isa_str = "DEFAULT";
  std::string output_bin_name = "";
  std::string target_config_path = "";
  std::string output_format = "bin";
  // Default ON (as "auto"): the split is the packaging that measures better on
  // every lane -- smaller resident plane, faster init, byte-identical output.
  // A CLI flag / config key, deliberately not an env var: packaging is a
  // property of the artifact, and an env var leaves no trace in the package.
  SidecarMode sidecar_mode = SidecarMode::AUTO;
  bool sidecar_mode_from_cli = false;

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
      output_dir = argv[++i];
    } else if (arg == "--fc_dtype" && i + 1 < argc) {
      fc_dtype_str = argv[++i];
    } else if (arg == "--embd_dtype" && i + 1 < argc) {
      embd_dtype_str = argv[++i];
    } else if (arg == "--lmhead_dtype" && i + 1 < argc) {
      lmhead_dtype_str = argv[++i];
    } else if (arg == "--isa" && i + 1 < argc) {
      isa_str = argv[++i];
    } else if (arg == "--output_bin" && i + 1 < argc) {
      output_bin_name = argv[++i];
    } else if (arg == "--output_format" && i + 1 < argc) {
      output_format = toLower(argv[++i]);
      if (output_format != "bin" && output_format != "safetensors") {
        std::cerr << "Unknown output format: " << output_format
                  << " (expected 'bin' or 'safetensors')\n";
        return EXIT_FAILURE;
      }
    } else if (arg == "--config" && i + 1 < argc) {
      target_config_path = argv[++i];
    } else if (arg == "--sidecar" && i + 1 < argc) {
      const std::string mode_arg = argv[++i];
      try {
        sidecar_mode = strToSidecarMode(mode_arg);
      } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
      }
      sidecar_mode_from_cli = true;
    } else if (arg == "--no-sidecar") {
      sidecar_mode = SidecarMode::OFF;
      sidecar_mode_from_cli = true;
    } else if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      printUsage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  try {
    // =========================================================================
    // Step 1: Load source configurations
    // =========================================================================
    std::cout << "==========================================================\n";
    std::cout << "  NNTrainer CausalLM Quantization Utility\n";
    std::cout << "==========================================================\n";
    std::cout << "[1/5] Loading configurations from: " << model_path << "\n";

    json cfg = causallm::LoadJsonFile(model_path + "/config.json");
    json generation_cfg =
      causallm::LoadJsonFile(model_path + "/generation_config.json");
    json nntr_cfg = causallm::LoadJsonFile(model_path + "/nntr_config.json");

    // If a target config is specified, read dtypes from it
    if (!target_config_path.empty()) {
      std::cout << "  Using target config: " << target_config_path << "\n";
      json target_cfg = causallm::LoadJsonFile(target_config_path);
      if (target_cfg.contains("fc_layer_dtype"))
        fc_dtype_str = target_cfg["fc_layer_dtype"].get<std::string>();
      if (target_cfg.contains("embedding_dtype"))
        embd_dtype_str = target_cfg["embedding_dtype"].get<std::string>();
      if (target_cfg.contains("lmhead_dtype"))
        lmhead_dtype_str = target_cfg["lmhead_dtype"].get<std::string>();
      if (target_cfg.contains("model_file_name") && output_bin_name.empty())
        output_bin_name = target_cfg["model_file_name"].get<std::string>();
      // The target config is the file the graph reads, so it also carries the
      // packaging decision. An explicit CLI flag still wins over it.
      if (!sidecar_mode_from_cli) {
        if (target_cfg.contains("sidecar")) {
          const auto &value = target_cfg["sidecar"];
          sidecar_mode =
            value.is_boolean()
              ? (value.get<bool>() ? SidecarMode::ON : SidecarMode::OFF)
              : strToSidecarMode(value.get<std::string>());
        } else if (target_cfg.contains("embedding_file_name") ||
                   target_cfg.contains("ple_file_name")) {
          // A target config that already names the manifests is asking for the
          // split, not merely permitting it.
          sidecar_mode = SidecarMode::ON;
        }
      }
    }

    // A source package whose own config names the manifests is ALREADY split:
    // its lookup layers read a sidecar and hold no in-bin weight, so its bin is
    // slim and every later record in it sits at a different offset than the
    // graph expects. Loading it would misalign silently and the quantized
    // output would be garbage that still exits 0.
    for (const auto &candidate : sidecar_candidates) {
      NNTR_THROW_IF(nntr_cfg.contains(candidate.config_key), std::runtime_error)
        << "source package " << model_path
        << " is already sidecar-split (its nntr_config.json carries '"
        << candidate.config_key
        << "'). Quantize from the single-file source package instead.";
    }

    // Default lmhead_dtype to embd_dtype if not specified
    if (lmhead_dtype_str.empty())
      lmhead_dtype_str = embd_dtype_str;

    // Parse target ISA
    ml::train::ISA target_isa = strToISA(isa_str);

    // Parse target data types
    DataType fc_dtype = strToDataType(fc_dtype_str);
    DataType embd_dtype = strToDataType(embd_dtype_str);
    DataType lmhead_dtype = strToDataType(lmhead_dtype_str);

    // Validate source model is FP32
    std::string src_tensor_type =
      nntr_cfg["model_tensor_type"].get<std::string>();
    if (src_tensor_type != "FP32-FP32") {
      std::cerr << "[WARNING] Source model_tensor_type is '" << src_tensor_type
                << "', not 'FP32-FP32'.\n"
                << "  Quantization from non-FP32 models may produce unexpected "
                   "results.\n";
    }

    // Setup output directory
    if (output_dir.empty())
      output_dir = model_path;
    std::filesystem::create_directories(output_dir);

    // Determine output filename
    std::string original_bin = nntr_cfg["model_file_name"].get<std::string>();
    if (output_bin_name.empty()) {
      output_bin_name =
        generateOutputBinName(original_bin, dataTypeToStr(fc_dtype),
                              dataTypeToStr(embd_dtype), isa_str);
    }
    if (output_format == "safetensors") {
      // The output format is decided by the file extension on save, so make
      // sure the generated/explicit name ends with ".safetensors".
      const std::string bin_ext = ".bin";
      auto pos = output_bin_name.rfind(bin_ext);
      if (pos != std::string::npos &&
          pos + bin_ext.size() == output_bin_name.size())
        output_bin_name = output_bin_name.substr(0, pos) + ".safetensors";
      else if (output_bin_name.find(".safetensors") == std::string::npos)
        output_bin_name += ".safetensors";
    }

    std::string src_weight_path = model_path + "/" + original_bin;
    std::string dst_weight_path = output_dir + "/" + output_bin_name;

    int num_layers = cfg["num_hidden_layers"].get<int>();
    std::string architecture =
      cfg["architectures"].get<std::vector<std::string>>()[0];

    std::cout << "  Architecture: " << architecture << "\n";
    std::cout << "  Num layers:   " << num_layers << "\n";
    std::cout << "  Source:       " << src_weight_path << "\n";
    std::cout << "  Target:       " << dst_weight_path << "\n";
    std::cout << "  FC dtype:     " << dataTypeToStr(fc_dtype) << "\n";
    std::cout << "  Embed dtype:  " << dataTypeToStr(embd_dtype) << "\n";
    std::cout << "  LMHead dtype: " << dataTypeToStr(lmhead_dtype) << "\n";
    std::cout << "  Target ISA:   " << isaToStr(target_isa) << "\n";
    std::cout << "\n";

    // =========================================================================
    // Step 2: Register models & create model instance
    // =========================================================================
    std::cout << "[2/5] Creating and initializing model...\n";

    registerAllModels();

    if (nntr_cfg.contains("model_type")) {
      std::string model_type = nntr_cfg["model_type"].get<std::string>();
      architecture = resolve_architecture(model_type, architecture);
    }

    // Resolve paths in nntr_cfg against the model directory.
    // Relative paths are anchored to model_path (existing behaviour).
    // Absolute paths that don't exist in the current environment (e.g. a path
    // baked in on the build host but running on an Android device) fall back to
    // the bare filename resolved against model_path, so the file is found as
    // long as it lives next to nntr_config.json.
    for (const char *key : {"module_config_path", "tokenizer_file"}) {
      if (!nntr_cfg.contains(key))
        continue;
      std::filesystem::path p = nntr_cfg[key].get<std::string>();
      if (p.is_relative()) {
        nntr_cfg[key] = (std::filesystem::path(model_path) / p).string();
      } else if (!std::filesystem::exists(p)) {
        nntr_cfg[key] =
          (std::filesystem::path(model_path) / p.filename()).string();
      }
    }

    auto model = causallm::Factory::Instance().create(architecture, cfg,
                                                      generation_cfg, nntr_cfg);
    if (!model) {
      throw std::runtime_error("Failed to create model for architecture: " +
                               architecture);
    }

    model->initialize();
    std::cout << "  Model initialized successfully.\n";

    // The dtype map depends only on the config, and the sidecar plan depends
    // only on it and on the compiled graph -- neither needs a single weight.
    // Deciding both HERE means a refusal costs a graph build instead of a
    // multi-gigabyte FP32 load.
    bool include_lmhead = true;
    if (nntr_cfg.contains("model_type") &&
        toLower(nntr_cfg["model_type"].get<std::string>()) == "embedding") {
      include_lmhead = false;
    }

    auto layer_dtype_map = buildLayerDtypeMap(num_layers, fc_dtype, embd_dtype,
                                              lmhead_dtype, include_lmhead);
    addSentenceTransformerLayerDtypes(layer_dtype_map, nntr_cfg, model_path,
                                      fc_dtype);

    // -------------------------------------------------------------------
    // Plan the mmap sidecar split.
    //
    // Eligibility is read off the COMPILED GRAPH, never re-derived from the
    // config: a tied table is a "tie_word_embeddings" node and an untied one
    // is an "embedding_layer", and constructModel() (base Transformer:
    // tie_word_embeddings; Gemma4: tie_word_embeddings && !lmhead_untie) is
    // the sole owner of that rule. Asking the graph is what keeps the packager
    // from drifting away from the model that will later load the package.
    // -------------------------------------------------------------------
    std::vector<SidecarPlan> sidecar_plan;
    std::vector<std::string> sidecar_blockers;

    if (sidecar_mode != SidecarMode::OFF && output_format != "bin") {
      sidecar_blockers.emplace_back(
        "output container is '" + output_format +
        "': the split is defined only for the .bin container");
    } else if (sidecar_mode != SidecarMode::OFF) {
      const auto tables = model->list_embedding_tables();
      const std::string output_base = stripWeightExtension(output_bin_name);

      for (const auto &candidate : sidecar_candidates) {
        const auto found =
          std::find_if(tables.begin(), tables.end(),
                       [&](const causallm::Transformer::EmbeddingTable &t) {
                         return t.name == candidate.layer_name;
                       });
        if (found == tables.end())
          continue; // this architecture has no such table -- not a blocker

        if (found->type != "embedding_layer") {
          // Name the two knobs precisely: config.json's tie_word_embeddings is
          // what EVERY model keys embedding-0 off (transformer.cpp:408), while
          // lmhead_untie reaches embedding-0 only in Gemma4
          // (gemma4_causallm.cpp:270 -- the base graph consults it for the
          // head alone). Recommending the wrong one costs a full FP32 load.
          sidecar_blockers.emplace_back(
            std::string(candidate.layer_name) + " is a '" + found->type +
            "' node: the sidecar path requires an UNTIED lookup table. Start "
            "from a checkpoint whose config.json has "
            "\"tie_word_embeddings\": false (works for every model), or -- for "
            "Gemma4, the only architecture whose embedding-0 honours it -- set "
            "\"lmhead_untie\": true in nntr_config.json.");
          continue;
        }
        if (found->rows == 0 || found->size == 0) {
          sidecar_blockers.emplace_back(std::string(candidate.layer_name) +
                                        " holds no weight to split");
          continue;
        }

        const auto dtype_it = layer_dtype_map.find(candidate.layer_name);
        const DataType table_dtype = (dtype_it != layer_dtype_map.end())
                                       ? dtype_it->second
                                       : DataType::NONE;
        const size_t row_bytes = ggmlRowBytes(table_dtype, found->size);
        if (row_bytes == 0) {
          sidecar_blockers.emplace_back(
            std::string(candidate.layer_name) + " is saved as " +
            (table_dtype == DataType::NONE ? std::string("unquantized (as-is)")
                                           : dataTypeToStr(table_dtype)) +
            " with out_dim " + std::to_string(found->size) +
            ": the sidecar manifest loader reads q4_0 (out_dim % 32 == 0) and "
            "q6_k (out_dim % 256 == 0) row blocks only -- use --embd_dtype "
            "Q4_0 or Q6_K.");
          continue;
        }

        sidecar_plan.push_back({candidate.layer_name, candidate.config_key,
                                output_base + candidate.suffix + ".json",
                                output_base + candidate.suffix + ".bin",
                                table_dtype, found->rows, found->size,
                                row_bytes});
      }
    }

    // Mode ON means "this package must be split": refuse rather than quietly
    // emitting the bigger single-file form under a name that promised the
    // smaller one.
    if (sidecar_mode == SidecarMode::ON &&
        (sidecar_plan.empty() || !sidecar_blockers.empty())) {
      std::string message =
        "--sidecar on was requested, but this model cannot be packaged with "
        "mmap sidecars:";
      for (const auto &blocker : sidecar_blockers)
        message += "\n    - " + blocker;
      if (sidecar_blockers.empty())
        message += "\n    - the model has no embedding-0 / per-layer-embedding "
                   "lookup table to split";
      message += "\n  Re-run with --no-sidecar to emit the single-file "
                 "package.";
      throw std::runtime_error(message);
    }

    // =========================================================================
    // Step 3: Load FP32 weights
    // =========================================================================
    std::cout << "[3/5] Loading FP32 weights from: " << src_weight_path << "\n";
    model->load_weight(src_weight_path);
    std::cout << "  Weights loaded successfully.\n";

    // =========================================================================
    // Step 4: Save quantized weights
    // =========================================================================
    std::cout << "[4/5] Quantizing and saving weights to: " << dst_weight_path
              << "\n";

    std::cout << "  Layer dtype mapping (" << layer_dtype_map.size()
              << " layers targeted):\n";
    for (const auto &[name, dt] : layer_dtype_map) {
      std::cout << "    " << name << " -> " << dataTypeToStr(dt) << "\n";
    }

    if (sidecar_plan.empty()) {
      if (sidecar_mode != SidecarMode::OFF) {
        std::cout << "  [sidecar] NOT APPLIED -- emitting the single-file "
                     "package:\n";
        if (sidecar_blockers.empty())
          std::cout << "    - no embedding-0 / per-layer-embedding lookup "
                       "table to split\n";
        for (const auto &blocker : sidecar_blockers)
          std::cout << "    - " << blocker << "\n";
      }
      model->save_weight(dst_weight_path, DataType::NONE, layer_dtype_map,
                         target_isa);
    } else {
      for (const auto &blocker : sidecar_blockers)
        std::cout << "  [sidecar] skipped: " << blocker << "\n";

      std::map<std::string, std::string> routed;
      for (const auto &plan : sidecar_plan)
        routed[plan.layer_name] = output_dir + "/" + plan.payload_name;

      const auto written = model->save_weight_split(
        dst_weight_path, DataType::NONE, layer_dtype_map, routed, target_isa);

      for (const auto &plan : sidecar_plan) {
        // The manifest loader derives in_dim from payload_size / row_bytes and
        // then rejects a "rows" that disagrees. Check that arithmetic HERE,
        // against the bytes actually written, so a package can never be
        // shipped with a manifest its own loader will refuse.
        const uint64_t expected =
          static_cast<uint64_t>(plan.rows) * plan.row_bytes;
        const uint64_t actual = written.at(plan.layer_name);
        NNTR_THROW_IF(actual != expected, std::runtime_error)
          << "sidecar payload " << plan.payload_name << " is " << actual
          << " bytes but rows(" << plan.rows << ") * row_bytes("
          << plan.row_bytes << ") = " << expected;

        json manifest;
        manifest["datatype"] = sidecarDatatypeName(plan.dtype);
        manifest["lut-path"] = plan.payload_name;
        manifest["rows"] = plan.rows;
        manifest["size"] = plan.size;

        const std::string manifest_path = output_dir + "/" + plan.manifest_name;
        std::ofstream manifest_out(manifest_path);
        NNTR_THROW_IF(!manifest_out.is_open(), std::runtime_error)
          << "Failed to open sidecar manifest: " << manifest_path;
        manifest_out << manifest.dump(4) << std::endl;
        manifest_out.close();

        std::cout << "  [sidecar] " << plan.layer_name << " -> "
                  << plan.payload_name << " (" << plan.rows << " rows x "
                  << plan.row_bytes << " B = " << actual << " B), manifest "
                  << plan.manifest_name << "\n";
      }
    }

    // Report file size
    auto src_size = std::filesystem::file_size(src_weight_path);
    auto dst_size = std::filesystem::file_size(dst_weight_path);
    for (const auto &plan : sidecar_plan)
      dst_size +=
        std::filesystem::file_size(output_dir + "/" + plan.payload_name);
    double ratio = static_cast<double>(dst_size) / src_size * 100.0;

    std::cout << "  Source size:  " << (src_size / (1024 * 1024)) << " MB\n";
    std::cout << "  Output size:  " << (dst_size / (1024 * 1024)) << " MB";
    if (!sidecar_plan.empty())
      std::cout << " (main bin "
                << (std::filesystem::file_size(dst_weight_path) / (1024 * 1024))
                << " MB + sidecars)";
    std::cout << "\n";
    std::cout << "  Compression:  " << std::fixed << std::setprecision(1)
              << ratio << "%\n";

    // =========================================================================
    // Step 5: Generate new nntr_config.json
    // =========================================================================
    std::cout << "[5/5] Generating nntr_config.json...\n";

    json new_nntr_cfg = nntr_cfg;
    new_nntr_cfg["model_file_name"] = output_bin_name;
    new_nntr_cfg["fc_layer_dtype"] = dataTypeToStr(fc_dtype);
    new_nntr_cfg["embedding_dtype"] = dataTypeToStr(embd_dtype);
    new_nntr_cfg["lmhead_dtype"] = dataTypeToStr(lmhead_dtype);
    new_nntr_cfg["model_tensor_type"] =
      buildModelTensorType(dataTypeToStr(fc_dtype));

    // Point the runtime at the manifests. Every key is (re-)decided here, so a
    // config inherited from a split source can never keep a stale manifest
    // path for a table this run wrote back into the bin.
    for (const auto &candidate : sidecar_candidates)
      new_nntr_cfg.erase(candidate.config_key);
    for (const auto &plan : sidecar_plan)
      new_nntr_cfg[plan.config_key] = plan.manifest_name;

    std::string output_config_path = output_dir + "/nntr_config.json";

    // If output is same dir and we'd overwrite, save as
    // nntr_config_quantized.json
    if (output_dir == model_path) {
      output_config_path = output_dir + "/nntr_config_quantized.json";
    }

    std::ofstream config_out(output_config_path);
    if (!config_out.is_open()) {
      throw std::runtime_error("Failed to open output config: " +
                               output_config_path);
    }
    config_out << new_nntr_cfg.dump(4) << std::endl;
    config_out.close();

    std::cout << "  Config saved to: " << output_config_path << "\n";

    // When writing to a separate directory, copy the auxiliary files the
    // runtime needs (config.json / generation_config.json are required by
    // CausalLM; tokenizer files are needed for generation) so the output
    // directory is self-contained and runnable on its own.
    if (output_dir != model_path) {
      const char *aux_files[] = {"config.json",
                                 "generation_config.json",
                                 "tokenizer.json",
                                 "tokenizer_config.json",
                                 "special_tokens_map.json",
                                 "vocab.json",
                                 "merges.txt",
                                 "modules.json"};
      for (const char *fname : aux_files) {
        std::filesystem::path src = std::filesystem::path(model_path) / fname;
        if (!std::filesystem::exists(src))
          continue;
        std::filesystem::copy_file(
          src, std::filesystem::path(output_dir) / fname,
          std::filesystem::copy_options::overwrite_existing);
        std::cout << "  Copied " << fname << " to output directory\n";
      }
    }

    // =========================================================================
    // Done
    // =========================================================================
    std::cout << "\n";
    std::cout << "==========================================================\n";
    std::cout << "  Quantization complete!\n";
    std::cout << "==========================================================\n";
    std::cout << "\n";
    std::cout << "To run the quantized model:\n";
    if (output_dir == model_path) {
      std::cout << "  1. Rename nntr_config_quantized.json to "
                   "nntr_config.json\n";
      std::cout << "  2. nntr_causallm " << model_path << "\n";
    } else {
      std::cout << "  nntr_causallm " << output_dir << "\n\n";
    }

  } catch (const std::exception &e) {
    std::cerr << "\n[!] FATAL ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

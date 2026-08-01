// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    causal_lm_api.cpp
 * @date    21 Jan 2026
 * @brief   This is a C API for CausalLM application
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "causal_lm_api.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "callback_streamer.h"
#include "causal_lm.h"
#include "chat_template.h"
#include "json.hpp"
#include "model_config_internal.h"
#include "model_registry.h"
#include <factory.h>
#include <fstream>
#include <sys/stat.h>
#if !defined(_WIN32)
#include <unistd.h>
#endif

using json = nlohmann::json;

static std::unique_ptr<causallm::Transformer> g_model;
static std::mutex g_mutex;
// cancelModel() cannot take g_mutex, so this flag is the lock-free published
// loaded-model predicate. It is set only after g_model is ready to run.
static std::atomic<bool> g_initialized{false};
static std::mutex g_active_model_mutex;
static causallm::CausalLM *g_active_model = nullptr;
static std::string g_architecture = "";
static bool g_use_chat_template = false;
static bool g_verbose = false;
static std::string g_last_output = "";
// How long it took to build the model that is in memory. A load that reuses
// that model leaves this alone rather than reporting its own ~0 ms: the number
// getPerformanceMetrics() publishes is the cost of initialising the model being
// run, and overwriting it with 0 would tell a caller that polls after a
// re-load that the model came up for free.
static double g_initialization_duration_ms = 0.0;
static std::unique_ptr<causallm::ChatTemplate> g_chat_template;
// The loaded package's own chat render context ("chat_template_context" in
// nntr_config.json); per-call options are merged over it.
static json g_chat_template_context = json::object();
// The loaded package's generation_config.json "do_sample"; runModelWithOptions
// follows it unless the caller overrides it per call.
static bool g_do_sample_default = false;

/**
 * @brief What a load request asked for.
 *
 * Everything that can change the model a load produces lives here, so two
 * requests that compare equal are answered by the same weights:
 *  - @c source is the entry point, because the two describe a model in
 *    different terms and must not be read as each other's;
 *  - @c compute selects the nntrainer engine the graph is built for;
 *  - @c model_type and @c quant_type select the built-in configuration and
 *    the quantization suffix of the package directory (loadModel() only);
 *  - @c model_dir is the package directory itself, weakly canonicalised. It
 *    is what loadModelFromPath() was handed, and for loadModel() it is the
 *    directory the enums resolved to -- resolution is relative to the working
 *    directory, so the enums alone do not pin down which weights were read.
 *
 * Not here: the ::Config options, which change how a prompt is fed to a model
 * rather than which model is built, and are applied per run by setOptions().
 */
struct LoadRequest {
  /** @brief The entry point that made the request. */
  enum class Source { BuiltinType, PackageDirectory };

  Source source;
  BackendType compute;
  ModelType model_type;
  ModelQuantizationType quant_type;
  std::string model_dir;

  /** @brief Equality across every field that selects a model. */
  bool operator==(const LoadRequest &rhs) const {
    if (source != rhs.source || compute != rhs.compute ||
        model_dir != rhs.model_dir)
      return false;
    if (source == Source::PackageDirectory)
      return true; // the directory is the whole specification
    return model_type == rhs.model_type && quant_type == rhs.quant_type;
  }
};

/**
 * @brief The request g_model answers; empty when no model is loaded.
 *
 * Written only while holding g_mutex, and only together with g_model, so it
 * never describes a model that is not the one in memory.
 */
static std::optional<LoadRequest> g_loaded_request;

static std::map<std::string, std::string> g_model_path_map = {
  {"QWEN3-0.6B", "qwen3-0.6b"},
};

/**
 * @brief RegisteredModel
 */
struct RegisteredModel {
  std::string arch_name;
  ModelRuntimeConfig config;
};
static std::map<std::string, RegisteredModel> g_model_registry;
static std::map<std::string, ModelArchConfig> g_arch_config_map;

#ifdef ENABLE_TEST
namespace causal_lm_api_test {
using ActiveRunPublishHook = void (*)(void *);
using BeforeCancelRequestHook = void (*)(void *);
} // namespace causal_lm_api_test
#endif

namespace {

#ifdef ENABLE_TEST
causal_lm_api_test::ActiveRunPublishHook g_after_active_run_publish_hook =
  nullptr;
void *g_after_active_run_publish_user_data = nullptr;
causal_lm_api_test::BeforeCancelRequestHook g_before_cancel_request_hook =
  nullptr;
void *g_before_cancel_request_user_data = nullptr;
#endif

/** @brief RAII guard that tracks the currently active CausalLM run. */
class ActiveRunGuard {
public:
  explicit ActiveRunGuard(causallm::CausalLM *model) : model_(model) {
    if (model_ != nullptr) {
      std::lock_guard<std::mutex> lock(g_active_model_mutex);
      g_active_model = model_;
    }
  }

  ~ActiveRunGuard() {
    if (model_ != nullptr) {
      std::lock_guard<std::mutex> lock(g_active_model_mutex);
      if (g_active_model == model_)
        g_active_model = nullptr;
    }
  }

  ActiveRunGuard(const ActiveRunGuard &) = delete;
  ActiveRunGuard &operator=(const ActiveRunGuard &) = delete;

private:
  causallm::CausalLM *model_;
};

void notifyAfterActiveRunPublishForTest() {
#ifdef ENABLE_TEST
  auto *hook = g_after_active_run_publish_hook;
  if (hook != nullptr)
    hook(g_after_active_run_publish_user_data);
#endif
}

void notifyBeforeCancelRequestForTest() {
#ifdef ENABLE_TEST
  auto *hook = g_before_cancel_request_hook;
  if (hook != nullptr)
    hook(g_before_cancel_request_user_data);
#endif
}

void resolveNntrConfigPath(json &nntr_cfg, const std::string &key,
                           const std::string &model_dir_path) {
  if (!nntr_cfg.contains(key) || !nntr_cfg[key].is_string())
    return;

  std::filesystem::path path = nntr_cfg[key].get<std::string>();
  if (path.empty() || path.is_absolute())
    return;

  nntr_cfg[key] = (std::filesystem::path(model_dir_path) / path).string();
}

/**
 * @brief Package directory in a form two spellings of one directory share.
 *
 * Weak canonicalisation, so a package that has since been moved away still
 * normalises. A path that cannot be resolved at all falls back to its literal
 * spelling: the only consequence is that one directory named two ways reloads
 * instead of being reused, which costs time but never returns wrong weights.
 */
std::string canonical_model_dir(const std::string &model_dir) {
  try {
    return std::filesystem::weakly_canonical(std::filesystem::path(model_dir))
      .string();
  } catch (const std::exception &) {
    return model_dir;
  }
}

/**
 * @brief Whether the model already in memory answers @p request.
 * @note The caller must hold g_mutex; a load publishes g_model and
 *       g_loaded_request together under it, so the two cannot disagree here.
 */
bool loaded_model_answers(const LoadRequest &request) {
  return g_initialized.load(std::memory_order_acquire) && g_model != nullptr &&
         g_loaded_request.has_value() && *g_loaded_request == request;
}

} // namespace

#ifdef ENABLE_TEST
namespace causal_lm_api_test {

void setAfterActiveRunPublishHookForTest(ActiveRunPublishHook hook,
                                         void *user_data) {
  g_after_active_run_publish_hook = hook;
  g_after_active_run_publish_user_data = user_data;
}

void setBeforeCancelRequestHookForTest(BeforeCancelRequestHook hook,
                                       void *user_data) {
  g_before_cancel_request_hook = hook;
  g_before_cancel_request_user_data = user_data;
}

void setModelForTest(std::unique_ptr<causallm::Transformer> model,
                     const std::string &architecture) {
  std::lock_guard<std::mutex> lock(g_mutex);
  {
    std::lock_guard<std::mutex> active_lock(g_active_model_mutex);
    g_active_model = nullptr;
  }
  g_model = std::move(model);
  g_initialized.store(g_model != nullptr, std::memory_order_release);
  g_architecture = architecture;
  g_last_output.clear();
  g_chat_template.reset();
  // An injected model answers no load request until a test says which one, so
  // a later loadModel() cannot mistake it for the model it asked for.
  g_loaded_request.reset();
}

void resetForTest() {
  std::lock_guard<std::mutex> lock(g_mutex);
  {
    std::lock_guard<std::mutex> active_lock(g_active_model_mutex);
    g_active_model = nullptr;
  }
  g_model.reset();
  g_initialized.store(false, std::memory_order_release);
  g_loaded_request.reset();
  g_architecture.clear();
  g_use_chat_template = false;
  g_verbose = false;
  g_last_output.clear();
  g_initialization_duration_ms = 0.0;
  g_chat_template.reset();
  g_chat_template_context = json::object();
  g_do_sample_default = false;
  g_after_active_run_publish_hook = nullptr;
  g_after_active_run_publish_user_data = nullptr;
  g_before_cancel_request_hook = nullptr;
  g_before_cancel_request_user_data = nullptr;
}

std::string resolveNntrConfigPathForTest(const std::string &value,
                                         const std::string &model_dir_path) {
  json nntr_cfg;
  nntr_cfg["path"] = value;
  resolveNntrConfigPath(nntr_cfg, "path", model_dir_path);
  return nntr_cfg["path"].get<std::string>();
}

} // namespace causal_lm_api_test
#endif

/**
 * @brief Populate the factory with every runnable model, once.
 *
 * Delegates to causallm::registerAllModels() -- the registry that exists so
 * "alternative entry points (tests, SDK wrappers) register the same model set
 * without duplicating the list". This function used to carry its own hand-kept
 * list of nine architectures, which had already drifted: packages whose
 * architecture the runner loads fine failed here with "Unknown architecture"
 * purely because nobody added them in two places. Windows exclusions live in
 * the registry too, so they stay honoured.
 */
static void register_models() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    causallm::registerAllModels();

    // Register built-in configurations
    register_builtin_model_configs();
  });
}

static const char *get_model_name_from_type(ModelType type) {
  switch (type) {
  case CAUSAL_LM_MODEL_QWEN3_0_6B:
    return "QWEN3-0.6B";
  default:
    return nullptr;
  }
}

/**
 * @brief A rendered prompt together with the extents of the template's frame.
 *
 * The affixes travel with the bytes because the runner has to be able to
 * shorten the prompt without deleting the assistant-turn marker, and by the
 * time it sees the prompt the structure is gone -- it holds a string.
 */
struct RenderedPrompt {
  std::string text;
  causallm::PromptAffixes affixes;
};

/**
 * @brief Render the prompt through the loaded package's chat template.
 *
 * Delegates to the same seam the runner uses, so one prompt plus one model
 * package produces one prompt string on either front end.
 *
 * The per-architecture template table that used to live here is gone on
 * purpose. It was a second template implementation keyed on the architecture
 * string, and it could only ever be a guess: two packages of the same
 * architecture legitimately carry different templates, so the table silently
 * mis-templated them, and it hid a missing/broken template behind
 * plausible-looking markers. The template belongs to the model package.
 *
 * @throw std::exception from the renderer -- the caller gets an error instead
 *        of a quietly raw prompt.
 */
static RenderedPrompt apply_chat_template(const std::string &input,
                                          const json &call_context) {
  if (!g_chat_template) {
    std::cerr << "[Warning] No chat template in the loaded model package; "
                 "feeding the prompt as given."
              << std::endl;
    return RenderedPrompt{input, causallm::PromptAffixes{}};
  }

  json context = g_chat_template_context;
  if (call_context.is_object()) {
    for (auto it = call_context.begin(); it != call_context.end(); ++it)
      context[it.key()] = it.value();
  }

  // One request object renders the prompt and measures the template's own
  // frame around it, so the two cannot end up describing different renders.
  // This is what buildUserPrompt() does internally; it is spelled out only
  // because the runner needs the frame as well as the bytes.
  const json request = causallm::makeUserRequest(input, context);

  RenderedPrompt rendered;
  rendered.text = causallm::buildPrompt(g_chat_template.get(), request,
                                        causallm::PromptTemplateMode::Auto);
  rendered.affixes = causallm::promptAffixes(
    g_chat_template.get(), request, causallm::PromptTemplateMode::Auto);
  return rendered;
}

/**
 * @brief Parse a caller-supplied render-context JSON object.
 * @return the parsed object; an empty object when @p json_text is null/empty
 * @throw std::runtime_error when it is not parseable as a JSON object
 */
static json parse_chat_context(const char *json_text) {
  if (json_text == nullptr || json_text[0] == '\0')
    return json::object();

  json parsed = json::parse(json_text);
  if (!parsed.is_object())
    throw std::runtime_error("chat_context_json must be a JSON object");

  return parsed;
}

static std::string get_quantization_suffix(ModelQuantizationType type) {
  switch (type) {
  case CAUSAL_LM_QUANTIZATION_W4A32:
    return "-w4a32";
  case CAUSAL_LM_QUANTIZATION_W16A16:
    return "-w16a16";
  case CAUSAL_LM_QUANTIZATION_W8A16:
    return "-w8a16";
  case CAUSAL_LM_QUANTIZATION_W32A32:
    return "-w32a32";
  default: // W4A32 by default
    return "-w4a32";
  }
}

static std::string resolve_model_path(const std::string &model_key,
                                      ModelQuantizationType quant_type) {
  std::string path_upper = model_key;
  std::transform(path_upper.begin(), path_upper.end(), path_upper.begin(),
                 ::toupper);

  std::string base_dir_name = "";

  // 1. Try to find base directory name from map
  if (g_model_path_map.find(path_upper) != g_model_path_map.end()) {
    base_dir_name = g_model_path_map[path_upper];
  } else {
    // Fallback: use lowercased key as base dir name if not found in map
    // or just return empty? For restricted API, we should probably fail
    // earlier, but here we can return constructed path.
    base_dir_name = path_upper;
    std::transform(base_dir_name.begin(), base_dir_name.end(),
                   base_dir_name.begin(), ::tolower);
  }

  std::string model_path =
    "./models/" + base_dir_name + get_quantization_suffix(quant_type);

  return model_path;
}

/** @brief The request loadModel() answers for these arguments. */
static LoadRequest builtin_load_request(BackendType compute,
                                        ModelType modeltype,
                                        ModelQuantizationType quant_type,
                                        const std::string &model_name) {
  return LoadRequest{
    LoadRequest::Source::BuiltinType, compute, modeltype, quant_type,
    canonical_model_dir(resolve_model_path(model_name, quant_type))};
}

/** @brief The request loadModelFromPath() answers for these arguments. */
static LoadRequest package_load_request(BackendType compute,
                                        const std::string &model_dir) {
  // The enums say nothing about a package directory and operator== ignores
  // them for this source; they are filled only because C++ gives these enums
  // no "not applicable" enumerator.
  return LoadRequest{LoadRequest::Source::PackageDirectory, compute,
                     CAUSAL_LM_MODEL_QWEN3_0_6B, CAUSAL_LM_QUANTIZATION_UNKNOWN,
                     canonical_model_dir(model_dir)};
}

#ifdef ENABLE_TEST
namespace causal_lm_api_test {

void setLoadedBuiltinRequestForTest(BackendType compute, ModelType model_type,
                                    ModelQuantizationType quant_type) {
  const char *model_name = get_model_name_from_type(model_type);
  std::lock_guard<std::mutex> lock(g_mutex);
  g_loaded_request = builtin_load_request(
    compute, model_type, quant_type, model_name != nullptr ? model_name : "");
}

void setLoadedPackageRequestForTest(BackendType compute,
                                    const std::string &model_dir) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_loaded_request = package_load_request(compute, model_dir);
}

} // namespace causal_lm_api_test
#endif

static bool check_file_exists(const std::string &path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

static void validate_models() {
  std::cout << "[DEBUG] Validating model files..." << std::endl;
  // Iterate over all known model names in map
  for (auto const &[key, val] : g_model_path_map) {
    // We want to check for each Quantization Type if it exists
    // List of quant types to check: UNKNOWN (default), W4A32, W16A16, W32A32
    std::vector<ModelQuantizationType> quant_types = {
      CAUSAL_LM_QUANTIZATION_UNKNOWN, CAUSAL_LM_QUANTIZATION_W4A32,
      CAUSAL_LM_QUANTIZATION_W16A16, CAUSAL_LM_QUANTIZATION_W32A32};

    for (auto qt : quant_types) {
      std::string quant_suffix = get_quantization_suffix(qt);

      std::string lookup_key = key;
      if (qt != CAUSAL_LM_QUANTIZATION_UNKNOWN) {
        std::transform(quant_suffix.begin(), quant_suffix.end(),
                       quant_suffix.begin(), ::toupper); // "-W4A32"
        lookup_key += quant_suffix;
      }

      // Resolve path for this combination
      std::string resolved_path = resolve_model_path(key, qt);

      if (g_model_registry.find(lookup_key) != g_model_registry.end()) {
        // CASE 1: Configuration is registered in model_config.cpp
        // For these models, we only check if the binary weight file exists.
        // The configurations (config.json, etc.) are embedded in the library.
        RegisteredModel &rm = g_model_registry[lookup_key];
        std::string bin_file_name = rm.config.model_file_name;
        std::string full_path = resolved_path + "/" + bin_file_name;

        if (check_file_exists(full_path)) {
          std::cout << "  [OK] Reg Config: " << lookup_key << " -> "
                    << full_path << std::endl;
        } else {
          std::cout << "  [FAIL] Reg Config: " << lookup_key
                    << " -> Missing binary: " << full_path << std::endl;
        }

      } else {
        // CASE 2: No internal config, but model type exists (via map
        // iteration). For these models, we require external configuration files
        // (config.json, nntr_config.json) to be present in the directory.
        if (check_file_exists(resolved_path)) {
          bool has_config = check_file_exists(resolved_path + "/config.json");
          bool has_nntr =
            check_file_exists(resolved_path + "/nntr_config.json");

          if (has_config && has_nntr) {
            std::cout << "  [OK] External Config: " << lookup_key << " -> "
                      << resolved_path << std::endl;
            // Optional: Parse nntr_config to check bin
            try {
              json nntr =
                causallm::LoadJsonFile(resolved_path + "/nntr_config.json");
              if (nntr.contains("model_file_name")) {
                std::string bin = nntr["model_file_name"];
                if (check_file_exists(resolved_path + "/" + bin)) {
                  std::cout << "       (Binary confirmed: " << bin << ")"
                            << std::endl;
                } else {
                  std::cout << "       (MISSING BINARY: " << bin << ")"
                            << std::endl;
                }
              }
            } catch (...) {
            }
          } else {
            std::cout << "  [FAIL] External Config: " << lookup_key
                      << " -> Missing configs in " << resolved_path
                      << std::endl;
          }
        }
      }
    }
  }
}

ErrorCode setOptions(Config config) {
  // Currently no options are being handled
  g_use_chat_template = config.use_chat_template;
  g_verbose = config.verbose;
  if (config.debug_mode) {
    // Ensure models are registered so we can validate them
    register_models();
    validate_models();
  }
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode registerModelArchitecture(const char *arch_name,
                                    ModelArchConfig config) {
  if (arch_name == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  std::lock_guard<std::mutex> lock(g_mutex);
  std::string name(arch_name);
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  g_arch_config_map[name] = config;
  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode registerModel(const char *model_name, const char *arch_name,
                        ModelRuntimeConfig config) {
  if (model_name == nullptr || arch_name == nullptr)
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  std::lock_guard<std::mutex> lock(g_mutex);
  std::string name(model_name);
  std::transform(name.begin(), name.end(), name.begin(), ::toupper);

  std::string aname(arch_name);
  std::transform(aname.begin(), aname.end(), aname.begin(), ::toupper);

  g_model_registry[name] = {aname, config};
  return CAUSAL_LM_ERROR_NONE;
}

/**
 * @brief Shared tail of every load path.
 *
 * Both load entry points end here, so they cannot drift in what they set up --
 * above all the chat template and its render context, which the SDK has to
 * establish exactly as the command line runner does for the two front ends to
 * answer a prompt identically.
 *
 * @note repack_weight() runs here. The runner has always called it after
 *       load_weight() (and so does the SDK facade); the API load path used to
 *       skip it, which left backends whose weights need a repack running a
 *       different weight layout than the runner for the same package.
 *
 * @note Nothing the caller can observe is written until the replacement model
 *       is built and its weights are in memory. A load that fails -- by
 *       returning or by throwing out of initialize()/load_weight() -- leaves
 *       the previously loaded model loaded, runnable and described by its own
 *       chat template and identity. The cost is that both models are resident
 *       for the length of the swap, which is the price of not being left with
 *       neither.
 *
 * @param request what the caller asked for; recorded with the model it
 *                produced so a repeat of the same request can reuse it
 */
static ErrorCode finish_load(json &cfg, json &generation_cfg, json &nntr_cfg,
                             const std::string &model_dir_path,
                             const LoadRequest &request) {
  // Decoding policy is the package's to state, exactly as the runner reads it.
  const bool do_sample_default =
    generation_cfg.is_object() && generation_cfg.value("do_sample", false);

  // The chat template and its render defaults belong to the model package.
  json chat_template_context = causallm::chatTemplateContext(nntr_cfg);
  std::unique_ptr<causallm::ChatTemplate> chat_template;
  if (causallm::ChatTemplate::Exists(model_dir_path)) {
    try {
      chat_template = std::make_unique<causallm::ChatTemplate>(
        causallm::ChatTemplate::Load(model_dir_path));
      std::cout << "[Info] Chat template loaded from "
                << chat_template->sourcePath() << std::endl;
    } catch (const std::exception &e) {
      chat_template.reset();
      std::cerr << "[Warning] Failed to load chat template: " << e.what()
                << "; prompts will be fed as given." << std::endl;
    }
  } else {
    std::cout << "[Info] No chat template in " << model_dir_path
              << "; prompts will be fed as given." << std::endl;
  }

  // Construct weight file path
  std::string weight_file_name;
  if (nntr_cfg.contains("model_file_name")) {
    weight_file_name = nntr_cfg["model_file_name"].get<std::string>();
  } else {
    weight_file_name = "pytorch_model.bin"; // Default fallback if not specified
  }

  const std::string weight_file = model_dir_path + "/" + weight_file_name;

  // Architecture, in the runner's order of preference. Embedding packages are
  // out of scope for a text-generation API and are rejected rather than
  // silently mapped onto a generative architecture.
  std::string architecture;
  if (cfg.contains("architectures") && cfg["architectures"].is_array() &&
      !cfg["architectures"].empty()) {
    architecture = cfg["architectures"].get<std::vector<std::string>>()[0];
  } else if (cfg.contains("architecture") && cfg["architecture"].is_string()) {
    architecture = cfg["architecture"].get<std::string>();
  } else if (cfg.contains("model_type") && cfg["model_type"].is_string()) {
    architecture = cfg["model_type"].get<std::string>();
  } else {
    std::cerr << "config.json must contain 'architectures', 'architecture' or "
                 "'model_type'"
              << std::endl;
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  if (nntr_cfg.contains("model_type") && nntr_cfg["model_type"].is_string()) {
    std::string model_type = nntr_cfg["model_type"].get<std::string>();
    std::transform(model_type.begin(), model_type.end(), model_type.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (model_type == "embedding") {
      std::cerr << "embedding models are out of scope for this API"
                << std::endl;
      return CAUSAL_LM_ERROR_INVALID_PARAMETER;
    }
  }

  // Build the replacement to completion before touching the live model.
  auto next_model = causallm::Factory::Instance().create(
    architecture, cfg, generation_cfg, nntr_cfg);
  if (!next_model) {
    std::cerr << "Unknown architecture: " << architecture << std::endl;
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
  }

  next_model->initialize();
  next_model->load_weight(weight_file);
  next_model->repack_weight();

  // Runnable now, so publish it. g_mutex is held, so no run is in flight and
  // g_active_model is already null; clearing it keeps that an invariant of the
  // swap rather than an assumption about the caller.
  {
    std::lock_guard<std::mutex> active_lock(g_active_model_mutex);
    g_active_model = nullptr;
  }
  g_initialized.store(false, std::memory_order_release);
  g_model = std::move(next_model);
  g_architecture = architecture;
  g_chat_template = std::move(chat_template);
  g_chat_template_context = std::move(chat_template_context);
  g_do_sample_default = do_sample_default;
  g_loaded_request = request;
  g_initialized.store(true, std::memory_order_release);
  return CAUSAL_LM_ERROR_NONE;
}

/** @brief nntrainer engine name for a backend; empty when unsupported. */
static const char *engine_of(BackendType compute) {
  switch (compute) {
  case CAUSAL_LM_BACKEND_CPU:
    return "cpu";
  case CAUSAL_LM_BACKEND_GPU:
    return "gpu";
  default:
    return "";
  }
}

ErrorCode loadModelFromPath(BackendType compute, const char *model_dir) {
  if (model_dir == nullptr || model_dir[0] == '\0')
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;

  const char *engine = engine_of(compute);
  if (engine[0] == '\0') {
    std::cerr << "Unsupported backend for loadModelFromPath" << std::endl;
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  auto start_init = std::chrono::high_resolution_clock::now();

  register_models();

  std::lock_guard<std::mutex> lock(g_mutex);
  try {
    const std::string model_dir_path(model_dir);

    // Already serving this package: keep the weights that are in memory.
    const LoadRequest request = package_load_request(compute, model_dir_path);
    if (loaded_model_answers(request)) {
      std::cout << "[Info] " << model_dir_path
                << " is already loaded; reusing the model in memory."
                << std::endl;
      return CAUSAL_LM_ERROR_NONE;
    }

    if (!std::filesystem::exists(std::filesystem::path(model_dir_path) /
                                 "config.json") ||
        !std::filesystem::exists(std::filesystem::path(model_dir_path) /
                                 "nntr_config.json")) {
      std::cerr << model_dir_path
                << " must contain config.json and nntr_config.json"
                << std::endl;
      return CAUSAL_LM_ERROR_INVALID_PARAMETER;
    }

    // The engine name is latched the first time nntrainer reads it, so it must
    // be published before anything touches the global engine. An explicit
    // backend argument is authoritative for this process.
    if (!g_initialized.load(std::memory_order_acquire)) {
#ifdef _WIN32
      _putenv_s("NNTR_ENGINE", engine);
#else
      ::setenv("NNTR_ENGINE", engine, /*overwrite=*/1);
#endif
    }

    json cfg = causallm::LoadJsonFile(model_dir_path + "/config.json");
    json generation_cfg = json::object();
    const std::string generation_config_path =
      model_dir_path + "/generation_config.json";
    if (std::filesystem::exists(generation_config_path))
      generation_cfg = causallm::LoadJsonFile(generation_config_path);
    json nntr_cfg =
      causallm::LoadJsonFile(model_dir_path + "/nntr_config.json");

    resolveNntrConfigPath(nntr_cfg, "tokenizer_file", model_dir_path);
    resolveNntrConfigPath(nntr_cfg, "embedding_file_name", model_dir_path);
    resolveNntrConfigPath(nntr_cfg, "ple_file_name", model_dir_path);

    const ErrorCode ec =
      finish_load(cfg, generation_cfg, nntr_cfg, model_dir_path, request);
    if (ec != CAUSAL_LM_ERROR_NONE)
      return ec;

    auto finish_init = std::chrono::high_resolution_clock::now();
    g_initialization_duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(finish_init -
                                                            start_init)
        .count();
  } catch (const std::exception &e) {
    std::cerr << "Exception in loadModelFromPath: " << e.what() << std::endl;
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
  } catch (...) {
    std::cerr << "Unknown exception in loadModelFromPath" << std::endl;
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
  }

  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode loadModel(BackendType compute, ModelType modeltype,
                    ModelQuantizationType quant_type) {

  auto start_init = std::chrono::high_resolution_clock::now();

  const char *target_model_name = get_model_name_from_type(modeltype);
  if (target_model_name == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  // Ensure models/configs are registered (thread-safe via call_once)
  register_models();

  std::lock_guard<std::mutex> lock(g_mutex);
  try {

    // Already serving this model: keep the weights that are in memory. Nothing
    // below can produce a different model for the same request, so re-reading
    // them would cost a full load to arrive back where we already are -- and
    // the caller would lose a working model for the duration.
    const LoadRequest request =
      builtin_load_request(compute, modeltype, quant_type, target_model_name);
    if (loaded_model_answers(request)) {
      std::cout << "[Info] " << target_model_name
                << " is already loaded; reusing the model in memory."
                << std::endl;
      return CAUSAL_LM_ERROR_NONE;
    }

    // Check if it's a registered in-memory config
    std::string input_name = std::string(target_model_name);
    std::string input_name_upper = input_name;
    std::transform(input_name_upper.begin(), input_name_upper.end(),
                   input_name_upper.begin(), ::toupper);

    std::string quant_suffix = "";
    switch (quant_type) {
    case CAUSAL_LM_QUANTIZATION_W4A32:
      quant_suffix = "-W4A32";
      break;
    case CAUSAL_LM_QUANTIZATION_W16A16:
      quant_suffix = "-W16A16";
      break;
    case CAUSAL_LM_QUANTIZATION_W8A16:
      quant_suffix = "-W8A16";
      break;
    case CAUSAL_LM_QUANTIZATION_W32A32:
      quant_suffix = "-W32A32";
      break;
    default:
      break;
    }
    std::string lookup_name = input_name_upper + quant_suffix;

    json cfg;
    json generation_cfg;
    json nntr_cfg;
    // Both branches below read the package from the same directory, and it is
    // the directory the request identity was built from.
    const std::string model_dir_path =
      resolve_model_path(target_model_name, quant_type);

    // Check in-memory map first
    if (g_model_registry.find(lookup_name) != g_model_registry.end()) {
      // ------------------------------------------------------------------------
      // CASE 1: Model Configuration is Internal (Registered in
      // model_config.cpp)
      // ------------------------------------------------------------------------
      // In this case, we do NOT load config.json or nntr_config.json from disk.
      // We only locate the binary weight file.
      RegisteredModel &rm = g_model_registry[lookup_name];

      // Find architecture config
      if (g_arch_config_map.find(rm.arch_name) == g_arch_config_map.end()) {
        std::cerr << "Architecture '" << rm.arch_name
                  << "' not found for model '" << lookup_name << "'"
                  << std::endl;
        return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
      }
      ModelArchConfig &ac = g_arch_config_map[rm.arch_name];
      ModelRuntimeConfig &rc = rm.config;

      // Populate JSONs from Arch Struct
      cfg["vocab_size"] = ac.vocab_size;
      cfg["hidden_size"] = ac.hidden_size;
      cfg["intermediate_size"] = ac.intermediate_size;
      cfg["num_hidden_layers"] = ac.num_hidden_layers;
      cfg["num_attention_heads"] = ac.num_attention_heads;
      cfg["head_dim"] = ac.head_dim;
      cfg["num_key_value_heads"] = ac.num_key_value_heads > 0
                                     ? ac.num_key_value_heads
                                     : ac.num_attention_heads;
      cfg["max_position_embeddings"] = ac.max_position_embeddings;
      cfg["rope_theta"] = ac.rope_theta;
      cfg["rms_norm_eps"] = ac.rms_norm_eps;
      cfg["tie_word_embeddings"] = ac.tie_word_embeddings;
      if (ac.sliding_window != UINT_MAX) {
        cfg["sliding_window"] = ac.sliding_window;
      } else {
        cfg["sliding_window"] = nullptr;
      }
      cfg["sliding_window_pattern"] = ac.sliding_window_pattern;
      cfg["architectures"] = {std::string(ac.architecture)};

      if (ac.num_eos_token_ids > 0) {
        std::vector<unsigned int> eos_ids;
        for (unsigned int i = 0; i < ac.num_eos_token_ids; ++i)
          eos_ids.push_back(ac.eos_token_ids[i]);
        generation_cfg["eos_token_id"] = eos_ids;
      }
      generation_cfg["bos_token_id"] = ac.bos_token_id;

      // Populate JSONs from Runtime Struct
      generation_cfg["top_k"] = rc.top_k;
      generation_cfg["top_p"] = rc.top_p;
      generation_cfg["temperature"] = rc.temperature;
      generation_cfg["do_sample"] = false;

      nntr_cfg["batch_size"] = rc.batch_size;
      nntr_cfg["model_type"] = std::string(rc.model_type);
      nntr_cfg["model_tensor_type"] = std::string(rc.model_tensor_type);
      nntr_cfg["init_seq_len"] = rc.init_seq_len;
      nntr_cfg["max_seq_len"] = rc.max_seq_len;
      nntr_cfg["num_to_generate"] = rc.num_to_generate;
      nntr_cfg["fsu"] = rc.fsu;
      nntr_cfg["fsu_lookahead"] = rc.fsu_lookahead;
      nntr_cfg["embedding_dtype"] = std::string(rc.embedding_dtype);
      nntr_cfg["fc_layer_dtype"] = std::string(rc.fc_layer_dtype);
      nntr_cfg["model_file_name"] = std::string(rc.model_file_name);

      nntr_cfg["tokenizer_file"] = std::string(rc.tokenizer_file);
      if (strlen(rc.embedding_file_name) > 0)
        nntr_cfg["embedding_file_name"] = std::string(rc.embedding_file_name);
      if (strlen(rc.ple_file_name) > 0)
        nntr_cfg["ple_file_name"] = std::string(rc.ple_file_name);
      resolveNntrConfigPath(nntr_cfg, "tokenizer_file", model_dir_path);
      resolveNntrConfigPath(nntr_cfg, "embedding_file_name", model_dir_path);
      resolveNntrConfigPath(nntr_cfg, "ple_file_name", model_dir_path);

      if (strlen(rc.lmhead_dtype) > 0) {
        nntr_cfg["lmhead_dtype"] = std::string(rc.lmhead_dtype);
      }

      std::vector<unsigned int> bad_ids;
      for (unsigned int i = 0; i < rc.num_bad_word_ids; ++i)
        bad_ids.push_back(rc.bad_word_ids[i]);
      nntr_cfg["bad_word_ids"] = bad_ids;

    } else {
      // --------------------------------------------------
      // CASE 2: External Model Configuration (File-based)
      // --------------------------------------------------
      // The model type is registered (enum), but specific configuration for
      // this quantization is not in memory. We must load config.json and
      // nntr_config.json from the model directory

      // Load configuration files
      cfg = causallm::LoadJsonFile(model_dir_path + "/config.json");
      generation_cfg =
        causallm::LoadJsonFile(model_dir_path + "/generation_config.json");
      nntr_cfg = causallm::LoadJsonFile(model_dir_path + "/nntr_config.json");

      resolveNntrConfigPath(nntr_cfg, "tokenizer_file", model_dir_path);
      resolveNntrConfigPath(nntr_cfg, "embedding_file_name", model_dir_path);
      resolveNntrConfigPath(nntr_cfg, "ple_file_name", model_dir_path);
    }

    const ErrorCode ec =
      finish_load(cfg, generation_cfg, nntr_cfg, model_dir_path, request);
    if (ec != CAUSAL_LM_ERROR_NONE)
      return ec;

    auto finish_init = std::chrono::high_resolution_clock::now();
    auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      finish_init - start_init);
    g_initialization_duration_ms = init_duration.count();

  } catch (const std::exception &e) {
    std::cerr << "Exception in loadModel: " << e.what() << std::endl;
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
  } catch (...) {
    std::cerr << "Unknown exception in loadModel" << std::endl;
    return CAUSAL_LM_ERROR_MODEL_LOAD_FAILED;
  }

  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode runModelWithOptions(const char *inputTextPrompt,
                              const GenerationOptions *options,
                              const char **outputText) {
  if (!g_initialized.load(std::memory_order_acquire)) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }
  if (inputTextPrompt == nullptr || outputText == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  // NULL options = the documented defaults: template applied, package context.
  const bool apply_template =
    options == nullptr ? true : options->apply_chat_template;
  const char *context_json =
    options == nullptr ? nullptr : options->chat_context_json;
  const bool do_sample = (options == nullptr || options->do_sample < 0)
                           ? g_do_sample_default
                           : options->do_sample > 0;

  try {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_initialized.load(std::memory_order_acquire) || !g_model) {
      return CAUSAL_LM_ERROR_NOT_INITIALIZED;
    }

    auto *causal_lm_model = dynamic_cast<causallm::CausalLM *>(g_model.get());

    std::string input(inputTextPrompt);
    causallm::PromptAffixes affixes;

    if (apply_template) {
      RenderedPrompt rendered =
        apply_chat_template(input, parse_chat_context(context_json));
      input = std::move(rendered.text);
      affixes = rendered.affixes;
    }

    if (causal_lm_model != nullptr) {
      // Zero when the caller opted out: a raw prompt has no frame to protect,
      // and saying so is what keeps its truncation behaviour unchanged.
      causal_lm_model->setPromptAffixBytes(affixes.prefix_bytes,
                                           affixes.suffix_bytes);
      causal_lm_model->prepareForRun();
    }
    ActiveRunGuard active_run_guard(causal_lm_model);
    notifyAfterActiveRunPublishForTest();

    // We assume single batch request for this API.
    g_model->run(input, do_sample, "", "", g_verbose);

    g_last_output = ""; // Reset last output
    if (causal_lm_model) {
      g_last_output = causal_lm_model->getOutput(0);
    }

    *outputText = g_last_output.c_str();

  } catch (const std::exception &e) {
    std::cerr << "Exception in runModel: " << e.what() << std::endl;
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }

  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode runModel(const char *inputTextPrompt, const char **outputText) {
  // Unchanged contract: the global Config.use_chat_template still decides and
  // decoding stays greedy, so an existing caller sees no behaviour change.
  const GenerationOptions options{g_use_chat_template, nullptr,
                                  /*do_sample=*/0};
  return runModelWithOptions(inputTextPrompt, &options, outputText);
}

ErrorCode runModelStreaming(const char *inputTextPrompt,
                            const char **outputText,
                            CausalLmTokenCallback callback, void *user_data) {
  if (inputTextPrompt == nullptr || outputText == nullptr ||
      callback == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }
  if (!g_initialized.load(std::memory_order_acquire)) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  try {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_initialized.load(std::memory_order_acquire) || !g_model) {
      return CAUSAL_LM_ERROR_NOT_INITIALIZED;
    }

    auto *causal_lm_model = dynamic_cast<causallm::CausalLM *>(g_model.get());
    if (causal_lm_model == nullptr) {
      return CAUSAL_LM_ERROR_UNKNOWN;
    }

    std::string input(inputTextPrompt);
    causallm::PromptAffixes affixes;

    if (g_use_chat_template) {
      RenderedPrompt rendered = apply_chat_template(input, json::object());
      input = std::move(rendered.text);
      affixes = rendered.affixes;
    }
    causal_lm_model->setPromptAffixBytes(affixes.prefix_bytes,
                                         affixes.suffix_bytes);

    CallbackStreamer streamer;
    callback_streamer_init(&streamer, callback, user_data);
    causal_lm_model->setStreamer(&streamer.base);

    struct StreamerDetachGuard {
      causallm::CausalLM *model;
      ~StreamerDetachGuard() { model->setStreamer(nullptr); }
    } detach_guard{causal_lm_model};

    causal_lm_model->prepareForRun();
    ActiveRunGuard active_run_guard(causal_lm_model);
    notifyAfterActiveRunPublishForTest();

    g_model->run(input, false, "", "", g_verbose);

    g_last_output = causal_lm_model->getOutput(0);
    *outputText = g_last_output.c_str();

  } catch (const std::exception &e) {
    std::cerr << "Exception in runModelStreaming: " << e.what() << std::endl;
    return CAUSAL_LM_ERROR_INFERENCE_FAILED;
  }

  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode cancelModel(void) {
  if (!g_initialized.load(std::memory_order_acquire)) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }

  std::lock_guard<std::mutex> active_lock(g_active_model_mutex);
  if (g_active_model != nullptr) {
    notifyBeforeCancelRequestForTest();
    g_active_model->requestStop();
  }

  return CAUSAL_LM_ERROR_NONE;
}

ErrorCode getPerformanceMetrics(PerformanceMetrics *metrics) {
  if (!g_initialized.load(std::memory_order_acquire)) {
    return CAUSAL_LM_ERROR_NOT_INITIALIZED;
  }
  if (metrics == nullptr) {
    return CAUSAL_LM_ERROR_INVALID_PARAMETER;
  }

  try {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_initialized.load(std::memory_order_acquire) || !g_model) {
      return CAUSAL_LM_ERROR_NOT_INITIALIZED;
    }

    if (!g_model->hasRun()) {
      return CAUSAL_LM_ERROR_INFERENCE_NOT_RUN;
    }
    auto internal_metrics = g_model->getPerformanceMetrics();
    metrics->prefill_tokens = internal_metrics.prefill_tokens;
    metrics->prefill_duration_ms = internal_metrics.prefill_duration_ms;
    metrics->generation_tokens = internal_metrics.generation_tokens;
    metrics->generation_duration_ms = internal_metrics.generation_duration_ms;
    metrics->total_duration_ms = internal_metrics.total_duration_ms;
    metrics->peak_memory_kb = internal_metrics.peak_memory_kb;

    // Overwrite init duration with the one measured in loadModel API
    metrics->initialization_duration_ms = g_initialization_duration_ms;

  } catch (const std::exception &e) {
    std::cerr << "Exception in getPerformanceMetrics: " << e.what()
              << std::endl;
    return CAUSAL_LM_ERROR_UNKNOWN;
  }

  return CAUSAL_LM_ERROR_NONE;
}

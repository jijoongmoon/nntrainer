// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   SFlareApi.cpp
 * @brief  Samsung FLARE API v2 — CausalLM-factory rebinding.
 *         The context owns one Factory-created model; architecture, dtype,
 *         tokenizer and generation defaults come from the model directory
 *         (config.json / nntr_config.json / chat template). The API layer
 *         only chooses the engine, applies validated environment bundles,
 *         and drives run()/KV-cache/streaming.
 * @date   14 July 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "SFlareApi.h"

#include "callback_streamer.h"

#include <causal_lm.h>
#include <chat_template.h>
#include <factory.h>
#include <model_registry.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace SFlareApi {

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

/**
 * @brief setenv shim; overwrite=0 keeps a caller-provided value intact so
 *        user environment always wins over profile bundles.
 */
void sflare_setenv(const char *key, const char *value, int overwrite) {
#ifdef _WIN32
  if (!overwrite && std::getenv(key) != nullptr)
    return;
  _putenv_s(key, value);
#else
  ::setenv(key, value, overwrite);
#endif
}

/**
 * @brief Rewrite a relative sidecar path in nntr_config.json to live under
 *        the model directory (same contract as main.cpp).
 */
void resolveNntrConfigPath(json &nntr_cfg, const std::string &key,
                           const std::string &model_path) {
  if (!nntr_cfg.contains(key) || !nntr_cfg[key].is_string())
    return;

  fs::path path = nntr_cfg[key].get<std::string>();
  if (path.empty() || path.is_absolute())
    return;

  nntr_cfg[key] = (fs::path(model_path) / path).string();
}

struct EnvKV {
  const char *key;
  const char *value;
};

/** [SFLARE_PHASE_TRACE=1] stderr timestamps (ms since first call) at load /
 *  first-execute phase boundaries — the init-latency dissection probe. */
void phase_trace(const char *what) {
  static const bool on = std::getenv("SFLARE_PHASE_TRACE") != nullptr;
  if (!on)
    return;
  static const auto t0 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
  std::fprintf(stderr, "[phase] %8.1f ms  %s\n", ms, what);
  std::fflush(stderr);
}

/** Common OpenCL kernel-path set (every CL device class). */
constexpr EnvKV kEnvClCommon[] = {{"NNTR_GPU_SVM_POOL", "1"},
                                  {"NNTR_MHA_GPU", "1"},
                                  {"NNTR_FC_GPU", "1"},
                                  {"NNTR_FC_INT8_GPU", "1"},
                                  {"NNTR_GPU_CLMEM_POOL", "1"}};

/** Intel Xe additions (XMX/DPAS GEMM + v8c buffer kernels). */
constexpr EnvKV kEnvClIntel[] = {{"NNTR_V8C_BUF", "1"}, {"NNTR_FC_XMX", "1"}};

/** Adreno bundle (image-attention path, validated device set). */
constexpr EnvKV kEnvClAdreno[] = {{"NNTR_NUM_THREADS", "4"},
                                  {"NNTR_FC_INT8_GPU", "1"},
                                  {"NNTR_MHA_GPU", "1"},
                                  {"NNTR_GPU_SVM_POOL", "1"},
                                  {"NNTR_KV_IMG_ATTN", "1"},
                                  {"NNTR_GPU_CLMEM_POOL", "1"}};

/** CUDA SAFE bundle (supported configuration). */
constexpr EnvKV kEnvCuda[] = {
  {"NNTR_CUDA_ROPE", "1"},     {"NNTR_CUDA_ATTN", "1"},
  {"NNTR_CUDA_KV_UVM", "1"},   {"NNTR_CUDA_GEGLU", "1"},
  {"NNTR_CUDA_ELTWISE", "1"},  {"NNTR_CUDA_QKNORM", "1"},
  {"NNTR_CUDA_FLASH_DECODE", "64"}, {"NNTR_CUDA_BLOCKQ", "1"},
  {"NNTR_FC_CUDA_CUBLAS", "1"}, {"NNTR_CUDA_PREWARM", "1"}};

#if defined(_WIN32)
/** CUDA A2 additions for WDDM (Windows). SAFE's UVM/coherence path
 * serializes under WDDM (~70/6.5 TPS on gauss4-side 1023tok); these are
 * exactly the accel levers the round-9/10 submit-batching discrimination
 * proved corruption-free there (M2B/ASYNC failed it and stay excluded).
 * Measured with this set: 4210/29.0 TPS, VRAM 1898MiB, goldens+kv-resume
 * green through this API. KV_DEV overrides the SAFE bundle's KV_UVM at
 * the consumer (both are set; the cuda KV manager prefers KV_DEV).
 * overwrite=0 like every bundle, so user env still wins — except
 * NNTR_CUDA_DEV_ACT which is presence-checked upstream (manager.h; known
 * issue, see HANDOFF §NEXT-2): setting DEV_ACT=0 cannot disable it. */
constexpr EnvKV kEnvCudaWddmA2[] = {{"NNTR_CUDA_DEV_ACT", "1"},
                                    {"NNTR_CUDA_VCOPY_PREFILL", "1"},
                                    {"NNTR_RMSNORM_CUDA_OFF", "all"},
                                    {"NNTR_CUDA_KV_DEV", "1"},
                                    {"NNTR_CUBLAS_WS_MB", "16"},
                                    {"NNTR_QS4CX_DECOMMIT", "1"}};
#endif

/** MemoryProfile::MINIMAL — steady-residency levers, TPS-neutral. */
constexpr EnvKV kEnvMinimalCuda[] = {{"NNTR_CUDA_I8_JIT", "1"},
                                     {"NNTR_QS4CX_HEAP_BYPASS", "1"},
                                     {"NNTR_CUDA_DROP_PLAIN", "1"}};
constexpr EnvKV kEnvMinimalClX86[] = {{"NNTR_QS4CX_HEAP_BYPASS", "1"},
                                      {"NNTR_V8C_DROP_PLAIN", "1"}};

/** MemoryProfile::PERFORMANCE — keep derived weight caches resident. */
constexpr EnvKV kEnvPerformanceCuda[] = {{"NNTR_CUDA_I8_JIT", "0"}};

template <size_t N>
void applyBundle(const EnvKV (&bundle)[N]) {
  for (const auto &kv : bundle)
    sflare_setenv(kv.key, kv.value, /*overwrite=*/0);
}

/** nntrainer engine string for a BackendType; empty = unsupported. */
std::string engineOf(BackendType compute) {
  switch (compute) {
  case BackendType::CPU:
    return "cpu";
  case BackendType::GPU:
  case BackendType::GPU_INTEL:
  case BackendType::GPU_ADRENO:
    return "gpu";
  case BackendType::GPU_NVIDIA:
    return "cuda";
  case BackendType::NPU:
  default:
    return "";
  }
}

/**
 * @brief The engine choice is process-wide (nntrainer contexts and the
 *        cached engine string are process-lifetime singletons); remember
 *        the first successful choice and refuse conflicting ones.
 */
std::mutex s_engine_mutex;
std::string s_process_engine;

} // namespace

SFlareContext::~SFlareContext() = default;

/**
 * @brief SFlareContext implementation bound to the CausalLM factory.
 */
class MySFlareContext : public SFlareContext {

public:
  MySFlareContext() = default;
  ~MySFlareContext() override = default;

  ErrorCode setSFlareOptions(SFlareConfig config) override {
    std::lock_guard<std::mutex> lock(run_mutex_);
    if (loaded_) {
      std::fprintf(stderr,
                   "[SFlare] setSFlareOptions after load is not supported; "
                   "create a new context\n");
      return ErrorCode::SFLARE_INVALID_CONFIG;
    }
    if (config.model_path == nullptr || config.model_path[0] == '\0') {
      std::fprintf(stderr, "[SFlare] model_path is required\n");
      return ErrorCode::SFLARE_INVALID_CONFIG;
    }

    std::string model_path(config.model_path);
    if (!fs::exists(fs::path(model_path) / "config.json") ||
        !fs::exists(fs::path(model_path) / "nntr_config.json")) {
      std::fprintf(stderr,
                   "[SFlare] %s must contain config.json and "
                   "nntr_config.json\n",
                   model_path.c_str());
      return ErrorCode::SFLARE_INVALID_CONFIG;
    }

    model_path_ = std::move(model_path);
    tokenizer_override_ =
      config.tokenizer_path ? std::string(config.tokenizer_path) : "";
    mem_profile_ = config.memory_profile;
    max_seq_override_ = config.max_seq_len;
    init_seq_override_ = config.init_seq_len;
    deterministic_ = config.deterministic;
    options_set_ = true;
    return ErrorCode::SFLARE_SUCCESS;
  }

  ErrorCode loadSFlareLLMModel(BackendType compute, bool enable_fsu,
                               ApplicationType app_type,
                               const char *lora_path) override {
    std::lock_guard<std::mutex> lock(run_mutex_);
    if (!options_set_)
      return ErrorCode::SFLARE_INVALID_CONFIG;
    if (loaded_)
      return ErrorCode::SFLARE_INVALID_CONFIG;
    if (app_type != ApplicationType::INSTRUCT || lora_path != nullptr) {
      std::fprintf(stderr, "[SFlare] only INSTRUCT is supported in v2\n");
      return ErrorCode::SFLARE_UNDEFIND;
    }
    if (enable_fsu) {
      std::fprintf(stderr, "[SFlare] FSU is not wired in v2; ignoring "
                           "enable_fsu (memory profile governs residency)\n");
    }

    const std::string engine = engineOf(compute);
    if (engine.empty()) {
      std::fprintf(stderr, "[SFlare] unsupported backend\n");
      return ErrorCode::SFLARE_UNDEFIND;
    }

    {
      std::lock_guard<std::mutex> engine_lock(s_engine_mutex);
      if (s_process_engine.empty()) {
        // Must precede the first Engine::Global()/initialize() in the
        // process: the engine string is latched on first read.
        sflare_setenv("NNTR_ENGINE", engine.c_str(), /*overwrite=*/1);
        // Must precede the engine bundles and the first Engine::Global():
        // the deterministic contract pins levers those would otherwise
        // auto-set (CUDA ASYNC) or default on (Intel FLASH_SG).
        if (deterministic_)
          sflare_setenv("NNTR_DETERMINISTIC", "1", /*overwrite=*/0);
        applyEnvBundles(compute, engine);
        s_process_engine = engine;
      } else if (s_process_engine != engine) {
        std::fprintf(stderr,
                     "[SFlare] engine is fixed to '%s' for this process; "
                     "cannot load on '%s'\n",
                     s_process_engine.c_str(), engine.c_str());
        return ErrorCode::SFLARE_INVALID_CONFIG;
      }
    }

    try {
      phase_trace("load: begin (env bundles applied)");
      causallm::registerAllModels();
      phase_trace("load: registerAllModels done");

      json cfg = causallm::LoadJsonFile(model_path_ + "/config.json");
      json generation_cfg = json::object();
      const std::string generation_config_path =
        model_path_ + "/generation_config.json";
      if (fs::exists(generation_config_path)) {
        generation_cfg = causallm::LoadJsonFile(generation_config_path);
      }
      json nntr_cfg = causallm::LoadJsonFile(model_path_ + "/nntr_config.json");
      resolveNntrConfigPath(nntr_cfg, "tokenizer_file", model_path_);
      resolveNntrConfigPath(nntr_cfg, "embedding_file_name", model_path_);
      resolveNntrConfigPath(nntr_cfg, "ple_file_name", model_path_);
      if (!tokenizer_override_.empty()) {
        nntr_cfg["tokenizer_file"] = tokenizer_override_;
      }
      // [seq override] SFlareConfig.max_seq_len / init_seq_len (>0) replace
      // the model-directory defaults: max_seq sizes the context/KV capacity,
      // init_seq the planned prefill activation plane. init is clamped to
      // max so a lone max override cannot strand init above it.
      if (max_seq_override_ > 0)
        nntr_cfg["max_seq_len"] = max_seq_override_;
      if (init_seq_override_ > 0)
        nntr_cfg["init_seq_len"] = init_seq_override_;
      if (nntr_cfg.contains("max_seq_len") &&
          nntr_cfg.contains("init_seq_len") &&
          nntr_cfg["init_seq_len"].get<unsigned int>() >
            nntr_cfg["max_seq_len"].get<unsigned int>()) {
        nntr_cfg["init_seq_len"] = nntr_cfg["max_seq_len"];
        std::fprintf(stderr, "[SFlare] init_seq_len clamped to max_seq_len (%u)\n",
                     nntr_cfg["max_seq_len"].get<unsigned int>());
      }
      if (max_seq_override_ > 0 || init_seq_override_ > 0)
        std::fprintf(stderr, "[SFlare] seq config: max_seq_len=%u init_seq_len=%u\n",
                     nntr_cfg["max_seq_len"].get<unsigned int>(),
                     nntr_cfg["init_seq_len"].get<unsigned int>());

      if (!nntr_cfg.contains("model_file_name")) {
        std::fprintf(stderr,
                     "[SFlare] nntr_config.json lacks model_file_name\n");
        return ErrorCode::SFLARE_INVALID_CONFIG;
      }
      const std::string weight_file =
        model_path_ + "/" + nntr_cfg["model_file_name"].get<std::string>();

      std::string architecture;
      if (cfg.contains("architectures") && cfg["architectures"].is_array() &&
          !cfg["architectures"].empty()) {
        architecture = cfg["architectures"].get<std::vector<std::string>>()[0];
      } else if (cfg.contains("architecture") &&
                 cfg["architecture"].is_string()) {
        architecture = cfg["architecture"].get<std::string>();
      } else if (cfg.contains("model_type") && cfg["model_type"].is_string()) {
        architecture = cfg["model_type"].get<std::string>();
      } else {
        return ErrorCode::SFLARE_INVALID_CONFIG;
      }
      if (nntr_cfg.contains("model_type") &&
          nntr_cfg["model_type"].is_string() &&
          nntr_cfg["model_type"].get<std::string>() == "embedding") {
        std::fprintf(stderr,
                     "[SFlare] embedding models are out of scope for the "
                     "SFlare LLM API\n");
        return ErrorCode::SFLARE_INVALID_CONFIG;
      }

      phase_trace("load: configs parsed");
      if (causallm::ChatTemplate::Exists(model_path_)) {
        try {
          chat_template_.emplace(causallm::ChatTemplate::Load(model_path_));
        } catch (const std::exception &e) {
          chat_template_.reset();
          std::fprintf(stderr, "[SFlare] chat template load failed: %s\n",
                       e.what());
        }
      }
      phase_trace("load: chat template loaded");

      auto start_init = std::chrono::high_resolution_clock::now();

      model_ = causallm::Factory::Instance().create(architecture, cfg,
                                                    generation_cfg, nntr_cfg);
      if (!model_) {
        std::fprintf(stderr, "[SFlare] unknown architecture: %s\n",
                     architecture.c_str());
        return ErrorCode::SFLARE_FAIL;
      }
      auto *clm = dynamic_cast<causallm::CausalLM *>(model_.get());
      if (clm == nullptr) {
        std::fprintf(stderr,
                     "[SFlare] architecture %s is not a CausalLM model\n",
                     architecture.c_str());
        model_.reset();
        return ErrorCode::SFLARE_FAIL;
      }

      default_num_to_generate_ = model_->getNumToGenerate();
      phase_trace("load: model object created (factory)");

      model_->initialize();
      phase_trace("load: initialize() done (graph compile + weight plane)");
      model_->load_weight(weight_file);
      phase_trace("load: load_weight() done (read + v8c prebuild)");
      model_->repack_weight();
      phase_trace("load: repack_weight() done");

      auto finish_init = std::chrono::high_resolution_clock::now();
      init_ms_ = std::chrono::duration<double, std::milli>(finish_init -
                                                           start_init)
                   .count();

      architecture_ = architecture;
      clm_.store(clm, std::memory_order_release);
      loaded_ = true;
    } catch (const std::exception &e) {
      std::fprintf(stderr, "[SFlare] load failed: %s\n", e.what());
      model_.reset();
      clm_.store(nullptr, std::memory_order_release);
      return ErrorCode::SFLARE_FAIL;
    }

    return ErrorCode::SFLARE_SUCCESS;
  }

  ErrorCode executeSFlareLLM(const char *input_utf8, char *output_utf8,
                             size_t output_size,
                             const GenParams *params) override {
    if (output_utf8 == nullptr || output_size == 0)
      return ErrorCode::SFLARE_INVALID_INPUT;
    std::string text;
    const ErrorCode ec = executeInternal(input_utf8, params, nullptr, &text);
    if (ec != ErrorCode::SFLARE_SUCCESS)
      return ec;
    return copyOut(text, output_utf8, output_size);
  }

  ErrorCode executeSFlareLLM(const char *input_utf8,
                             SFlareTokenCallback callback, void *user_data,
                             const GenParams *params) override {
    if (callback == nullptr)
      return ErrorCode::SFLARE_INVALID_INPUT;
    CallbackStreamer streamer;
    // SFlareTokenCallback and CausalLmTokenCallback share the exact C
    // signature int(const char*, void*).
    callback_streamer_init(&streamer, callback, user_data);
    return executeInternal(input_utf8, params, &streamer.base, nullptr);
  }

  ErrorCode executeSFlareLLM(const char *input_utf8, char *output_utf8,
                             size_t output_size, unsigned int prev_idx,
                             const char *kvcache_path,
                             const GenParams *params) override {
    if (output_utf8 == nullptr || output_size == 0 || kvcache_path == nullptr)
      return ErrorCode::SFLARE_INVALID_INPUT;
    if (!loaded_)
      return ErrorCode::SFLARE_NOT_LOADED;
    if (!fs::exists(kvcache_path)) {
      std::fprintf(stderr, "[SFlare] kv cache not found: %s\n", kvcache_path);
      return ErrorCode::SFLARE_INVALID_INPUT;
    }

    causallm::CausalLM *clm = clm_.load(std::memory_order_acquire);
    clm->setPrecomputedKVCache(kvcache_path, prev_idx);
    std::string text;
    const ErrorCode ec = executeInternal(input_utf8, params, nullptr, &text);
    // One-shot resume: disarm so later plain runs do not rewind onto the
    // saved cache with a stale position.
    clm->setPrecomputedKVCache("", 0);
    if (ec != ErrorCode::SFLARE_SUCCESS)
      return ec;
    return copyOut(text, output_utf8, output_size);
  }

  ErrorCode pauseSFlareLLM(unsigned int &idx) override {
    causallm::CausalLM *clm = clm_.load(std::memory_order_acquire);
    if (!loaded_ || clm == nullptr)
      return ErrorCode::SFLARE_NOT_LOADED;
    clm->requestStop();
    idx = static_cast<unsigned int>(clm->getKvLen());
    return ErrorCode::SFLARE_SUCCESS;
  }

  ErrorCode saveKVcache(const char *path) override {
    if (path == nullptr)
      return ErrorCode::SFLARE_INVALID_INPUT;
    if (!loaded_)
      return ErrorCode::SFLARE_NOT_LOADED;
    std::lock_guard<std::mutex> lock(run_mutex_);
    causallm::CausalLM *clm = clm_.load(std::memory_order_acquire);
    try {
      clm->save_kvcache(path, clm->getKvLen());
    } catch (const std::exception &e) {
      std::fprintf(stderr, "[SFlare] saveKVcache failed: %s\n", e.what());
      return ErrorCode::SFLARE_FAIL;
    }
    return ErrorCode::SFLARE_SUCCESS;
  }

  ErrorCode loadKVcache(const char *path, unsigned int token_len) override {
    if (path == nullptr)
      return ErrorCode::SFLARE_INVALID_INPUT;
    if (!loaded_)
      return ErrorCode::SFLARE_NOT_LOADED;
    if (!fs::exists(path)) {
      std::fprintf(stderr, "[SFlare] kv cache not found: %s\n", path);
      return ErrorCode::SFLARE_INVALID_INPUT;
    }
    std::lock_guard<std::mutex> lock(run_mutex_);
    clm_.load(std::memory_order_acquire)
      ->setPrecomputedKVCache(path, token_len);
    return ErrorCode::SFLARE_SUCCESS;
  }

  ErrorCode getPerformance(char *perf_csv, size_t size) override {
    if (perf_csv == nullptr || size == 0)
      return ErrorCode::SFLARE_INVALID_INPUT;
    SFlarePerformance p;
    const ErrorCode ec = getPerformance(p);
    if (ec != ErrorCode::SFLARE_SUCCESS)
      return ec;
    // Legacy CSV: tokenizer ms (not measured separately in v2 -> 0.00),
    // prefill tokens/ms/TPS, gen tokens/ms/TPS.
    const int n = std::snprintf(
      perf_csv, size, "0.00, %u, %.2f, %.2f, %u, %.2f, %.2f",
      p.prefill_tokens, p.prefill_duration_ms, p.prefill_tps,
      p.generation_tokens, p.generation_duration_ms, p.generation_tps);
    if (n < 0)
      return ErrorCode::SFLARE_FAIL;
    if (static_cast<size_t>(n) >= size)
      return ErrorCode::SFLARE_BUFFER_TOO_SMALL;
    return ErrorCode::SFLARE_SUCCESS;
  }

  ErrorCode getPerformance(SFlarePerformance &out) override {
    if (!loaded_)
      return ErrorCode::SFLARE_NOT_LOADED;
    std::lock_guard<std::mutex> lock(run_mutex_);
    if (!model_->hasRun())
      return ErrorCode::SFLARE_FAIL;
    const auto m = model_->getPerformanceMetrics();
    out.prefill_tokens = m.prefill_tokens;
    out.prefill_duration_ms = m.prefill_duration_ms;
    out.prefill_tps = m.prefill_duration_ms > 0.0
                        ? m.prefill_tokens / m.prefill_duration_ms * 1000.0
                        : 0.0;
    out.generation_tokens = m.generation_tokens;
    out.generation_duration_ms = m.generation_duration_ms;
    out.generation_tps =
      m.generation_duration_ms > 0.0
        ? m.generation_tokens / m.generation_duration_ms * 1000.0
        : 0.0;
    out.initialization_duration_ms = init_ms_;
    out.peak_memory_kb = m.peak_memory_kb;
    return ErrorCode::SFLARE_SUCCESS;
  }

private:
  /** Apply engine bundles + memory-profile levers (user env wins). */
  void applyEnvBundles(BackendType compute, const std::string &engine) {
    if (engine == "gpu") {
      if (compute == BackendType::GPU_ADRENO) {
        applyBundle(kEnvClAdreno);
      } else {
        applyBundle(kEnvClCommon);
        if (compute == BackendType::GPU_INTEL)
          applyBundle(kEnvClIntel);
        if (mem_profile_ == MemoryProfile::MINIMAL)
          applyBundle(kEnvMinimalClX86);
      }
    } else if (engine == "cuda") {
      applyBundle(kEnvCuda);
#if defined(_WIN32)
      // WDDM production set — see kEnvCudaWddmA2. The I8_JIT/HEAP_BYPASS/
      // DROP_PLAIN legs of the full A2 profile come from MINIMAL below.
      applyBundle(kEnvCudaWddmA2);
#endif
      if (mem_profile_ == MemoryProfile::MINIMAL)
        applyBundle(kEnvMinimalCuda);
      else if (mem_profile_ == MemoryProfile::PERFORMANCE)
        applyBundle(kEnvPerformanceCuda);
    }
  }

  /** Shared body of the execute overloads. */
  ErrorCode executeInternal(const char *input_utf8, const GenParams *params,
                            ::BaseStreamer *streamer, std::string *out_text) {
    if (input_utf8 == nullptr)
      return ErrorCode::SFLARE_INVALID_INPUT;
    if (!loaded_)
      return ErrorCode::SFLARE_NOT_LOADED;

    std::lock_guard<std::mutex> lock(run_mutex_);
    causallm::CausalLM *clm = clm_.load(std::memory_order_acquire);
    phase_trace("execute: begin");

    std::string text(input_utf8);
    const bool want_template = params == nullptr || params->apply_chat_template;
    if (want_template && chat_template_.has_value()) {
      try {
        json request =
          json::array({{{"role", "user"}, {"content", text}}});
        text = chat_template_->apply(request);
      } catch (const std::exception &e) {
        std::fprintf(stderr,
                     "[SFlare] chat template apply failed (%s); using raw "
                     "input\n",
                     e.what());
      }
    }

    const unsigned int max_new =
      (params != nullptr && params->max_new_tokens > 0)
        ? params->max_new_tokens
        : default_num_to_generate_;
    model_->setNumToGenerate(max_new);
    phase_trace("execute: template applied, run() entering "
                "(tokenize + prefill inside)");

    clm->prepareForRun();
    clm->setStreamer(streamer);
    struct StreamerDetachGuard {
      causallm::CausalLM *model;
      ~StreamerDetachGuard() { model->setStreamer(nullptr); }
    } detach_guard{clm};

    try {
      clm->run(text, params != nullptr && params->do_sample, "", "",
               /*log_output=*/false);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "[SFlare] run failed: %s\n", e.what());
      return ErrorCode::SFLARE_FAIL;
    }

    if (out_text != nullptr)
      *out_text = clm->getOutput(0);
    return ErrorCode::SFLARE_SUCCESS;
  }

  static ErrorCode copyOut(const std::string &text, char *out, size_t size) {
    const size_t n = std::min(text.size(), size - 1);
    std::memcpy(out, text.data(), n);
    out[n] = '\0';
    return n < text.size() ? ErrorCode::SFLARE_BUFFER_TOO_SMALL
                           : ErrorCode::SFLARE_SUCCESS;
  }

  std::mutex run_mutex_;
  std::string model_path_;
  std::string tokenizer_override_;
  MemoryProfile mem_profile_ = MemoryProfile::MINIMAL;
  bool deterministic_ = false;         /**< SFlareConfig.deterministic */
  unsigned int max_seq_override_ = 0;  /**< SFlareConfig.max_seq_len */
  unsigned int init_seq_override_ = 0; /**< SFlareConfig.init_seq_len */

  std::unique_ptr<causallm::Transformer> model_;
  std::atomic<causallm::CausalLM *> clm_{nullptr};
  std::optional<causallm::ChatTemplate> chat_template_;
  std::string architecture_;

  bool options_set_ = false;
  bool loaded_ = false;
  double init_ms_ = 0.0;
  unsigned int default_num_to_generate_ = 0;
};

SFlareContext *initSFlare(bool &registered) {
  try {
    causallm::registerAllModels();
    registered = true;
    return new MySFlareContext();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[SFlare] initSFlare failed: %s\n", e.what());
    registered = false;
    return nullptr;
  }
}

ErrorCode DestroySFlareContext(SFlareContext *context) {
  if (context == nullptr)
    return ErrorCode::SFLARE_INVALID_INPUT;
  delete context;
  return ErrorCode::SFLARE_SUCCESS;
}

} // namespace SFlareApi

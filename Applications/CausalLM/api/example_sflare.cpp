// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   example_sflare.cpp
 * @brief  Standalone SFlare API v2 consumer example -- the file shipped in
 *         the sflare_sdk package. Everything a client needs is SFlareApi.h
 *         (self-contained: <cstddef> + <string>) plus the libraries listed in
 *         the SDK's build_example script; the API reads every model-specific
 *         detail (architecture, dtype, tokenizer, chat template, generation
 *         defaults) from the model directory's config.json / nntr_config.json,
 *         and applies the validated per-backend environment bundle itself --
 *         set NO NNTR_* variables unless you mean to override.
 * @date   14 July 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Usage: example_sflare <model_dir> [cpu|gpu|intel|adreno|cuda]
 *                       ["prompt"] [--kv]
 *
 *   model_dir  directory holding config.json, nntr_config.json, the model
 *              .bin (+ optional sidecars) and tokenizer.json. Put
 *              chat_template.jinja there too -- without it the prompt is fed
 *              raw (no chat template) and instruction models drift.
 *   backend    intel = Intel Xe (XMX), cuda = NVIDIA, gpu = generic OpenCL,
 *              adreno = Qualcomm, cpu = host. First load latches the engine
 *              process-wide.
 *   --kv       demo the pause / saveKVcache / resume-from-file round trip.
 */

#include "SFlareApi.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

/** Streaming sink: the API delivers UTF-8 deltas as they decode. Return 0 to
 *  keep generating; any nonzero return cancels the generation. */
int on_delta(const char *delta, void * /*user_data*/) {
  std::fputs(delta, stdout);
  std::fflush(stdout);
  return 0;
}

SFlareApi::BackendType backend_of(const std::string &name) {
  if (name == "cpu")
    return SFlareApi::BackendType::CPU;
  if (name == "intel")
    return SFlareApi::BackendType::GPU_INTEL;
  if (name == "adreno")
    return SFlareApi::BackendType::GPU_ADRENO;
  if (name == "cuda")
    return SFlareApi::BackendType::GPU_NVIDIA;
  return SFlareApi::BackendType::GPU;
}

bool ok(SFlareApi::ErrorCode ec, const char *what) {
  if (ec != SFlareApi::ErrorCode::SFLARE_SUCCESS) {
    std::fprintf(stderr, "[example] %s failed (code %d)\n", what,
                 static_cast<int>(ec));
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <model_dir> [cpu|gpu|intel|adreno|cuda] "
                 "[\"prompt\"] [--kv]\n",
                 argv[0]);
    return 1;
  }
  const std::string model_dir = argv[1];
  const std::string backend = argc >= 3 ? argv[2] : "intel";
  const std::string prompt =
    argc >= 4 ? argv[3] : "What is the capital of South Korea?";
  const bool kv_demo = argc >= 5 && std::strcmp(argv[4], "--kv") == 0;

  // 1) Context + model directory. MemoryProfile defaults to MINIMAL (the
  //    memory-campaign levers); PERFORMANCE keeps derived weight caches
  //    resident instead.
  bool registered = false;
  SFlareApi::SFlareContext *ctx = SFlareApi::initSFlare(registered);
  if (ctx == nullptr) {
    std::fprintf(stderr, "[example] initSFlare failed\n");
    return 1;
  }
  SFlareApi::SFlareConfig config;
  config.model_path = model_dir.c_str();
  if (!ok(ctx->setSFlareOptions(config), "setSFlareOptions"))
    return 1;

  // 2) Load. This selects the engine (NNTR_ENGINE) and applies the
  //    per-backend env bundle with overwrite=0 -- anything you exported
  //    beforehand wins over the bundle.
  if (!ok(ctx->loadSFlareLLMModel(backend_of(backend)), "loadSFlareLLMModel"))
    return 1;

  // 3) Streaming generation. GenParams rides both execute overloads;
  //    apply_chat_template=false would feed the prompt raw.
  SFlareApi::GenParams params;
  params.max_new_tokens = 64;
  std::printf("--- streaming ---\n");
  if (!ok(ctx->executeSFlareLLM(prompt.c_str(), on_delta, nullptr, &params),
          "executeSFlareLLM(streaming)"))
    return 1;
  std::printf("\n");

  // 4) Performance counters (CSV line and struct form).
  SFlareApi::SFlarePerformance perf;
  if (ok(ctx->getPerformance(perf), "getPerformance")) {
    std::printf("prefill %u tok %.1f TPS | gen %u tok %.1f TPS | init %.0f ms "
                "| peak %zu KB\n",
                perf.prefill_tokens, perf.prefill_tps, perf.generation_tokens,
                perf.generation_tps, perf.initialization_duration_ms,
                perf.peak_memory_kb);
  }

  // 5) Optional: pause -> save the KV cache -> resume from file with new
  //    text appended. The resumed run continues the saved context.
  if (kv_demo) {
    std::printf("--- kv save / resume ---\n");
    unsigned int position = 0;
    if (!ok(ctx->pauseSFlareLLM(position), "pauseSFlareLLM"))
      return 1;
    const char *kv_path = "sflare_example_kv.bin";
    if (!ok(ctx->saveKVcache(kv_path), "saveKVcache"))
      return 1;
    std::printf("saved kv at token position %u\n", position);

    char output[8192];
    if (!ok(ctx->executeSFlareLLM(" Tell me more.", output, sizeof(output),
                                  position, kv_path, &params),
            "executeSFlareLLM(resume)"))
      return 1;
    std::printf("resume output: %s\n", output);
  }

  if (!ok(SFlareApi::DestroySFlareContext(ctx), "DestroySFlareContext"))
    return 1;
  std::printf("[example_sflare] done\n");
  return 0;
}

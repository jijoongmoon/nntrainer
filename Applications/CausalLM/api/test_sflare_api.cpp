// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   test_sflare_api.cpp
 * @brief  Reference consumer for SFlare API v2. Exercises the full
 *         lifecycle against a model directory (gauss4 is the release
 *         target): streaming generation, batch generation, CSV/struct
 *         performance, and the pause/saveKVcache/reRun round-trip.
 * @date   14 July 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Usage: test_sflare_api <model_dir> [cpu|gpu|intel|adreno|cuda]
 *                        ["prompt"] [--kv]
 */

#include "SFlareApi.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

int print_delta(const char *delta, void * /*user_data*/) {
  std::fputs(delta, stdout);
  std::fflush(stdout);
  return 0;
}

SFlareApi::BackendType backendOf(const std::string &name) {
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
    std::fprintf(stderr, "[test_sflare] %s failed (code %d)\n", what,
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
  const std::string backend = argc >= 3 ? argv[2] : "cpu";
  const std::string prompt =
    argc >= 4 ? argv[3] : "The capital of Korea is";
  const bool kv_smoke = argc >= 5 && std::strcmp(argv[4], "--kv") == 0;

  bool registered = false;
  SFlareApi::SFlareContext *ctx = SFlareApi::initSFlare(registered);
  if (ctx == nullptr) {
    std::fprintf(stderr, "[test_sflare] initSFlare failed\n");
    return 1;
  }

  SFlareApi::SFlareConfig config;
  config.model_path = model_dir.c_str();
  if (!ok(ctx->setSFlareOptions(config), "setSFlareOptions"))
    return 1;

  if (!ok(ctx->loadSFlareLLMModel(backendOf(backend)), "loadSFlareLLMModel"))
    return 1;

  SFlareApi::GenParams params;
  params.max_new_tokens = 32;

  std::printf("--- streaming ---\n");
  if (!ok(ctx->executeSFlareLLM(prompt.c_str(), print_delta, nullptr,
                                &params),
          "executeSFlareLLM(streaming)"))
    return 1;
  std::printf("\n");

  char perf_csv[256];
  if (ok(ctx->getPerformance(perf_csv, sizeof(perf_csv)),
         "getPerformance(csv)")) {
    std::printf("perf csv: %s\n", perf_csv);
  }
  SFlareApi::SFlarePerformance perf;
  if (ok(ctx->getPerformance(perf), "getPerformance(struct)")) {
    std::printf("prefill %u tok %.1f TPS | gen %u tok %.1f TPS | init %.0f "
                "ms | peak %zu KB\n",
                perf.prefill_tokens, perf.prefill_tps, perf.generation_tokens,
                perf.generation_tps, perf.initialization_duration_ms,
                perf.peak_memory_kb);
  }

  if (kv_smoke) {
    std::printf("--- kv save / reRun ---\n");
    unsigned int idx = 0;
    if (!ok(ctx->pauseSFlareLLM(idx), "pauseSFlareLLM"))
      return 1;
    const char *kv_path = "/tmp/sflare_kv_smoke.bin";
    if (!ok(ctx->saveKVcache(kv_path), "saveKVcache"))
      return 1;
    std::printf("saved kv at token position %u\n", idx);

    char output[8192];
    if (!ok(ctx->executeSFlareLLM(" It is a large city.", output,
                                  sizeof(output), idx, kv_path, &params),
            "executeSFlareLLM(reRun)"))
      return 1;
    std::printf("reRun output: %s\n", output);
  }

  if (!ok(SFlareApi::DestroySFlareContext(ctx), "DestroySFlareContext"))
    return 1;
  std::printf("[test_sflare] done\n");
  return 0;
}

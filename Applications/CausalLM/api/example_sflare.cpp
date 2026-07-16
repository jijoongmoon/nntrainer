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
 *                       ["prompt" | @textfile] [--kv]
 *                       [--max-seq N] [--init-seq N]
 *
 *   model_dir  directory holding config.json, nntr_config.json, the model
 *              .bin (+ optional sidecars) and tokenizer.json. Put
 *              chat_template.jinja there too -- without it the prompt is fed
 *              raw (no chat template) and instruction models drift.
 *   backend    intel = Intel Xe (XMX), cuda = NVIDIA, gpu = generic OpenCL,
 *              adreno = Qualcomm, cpu = host. First load latches the engine
 *              process-wide. (Intel dp4a = pass intel and export
 *              NNTR_FC_XMX=0 beforehand -- user env wins over the bundle.)
 *   prompt     literal text, or @file to summarize: the file's content is
 *              wrapped in a three-sentence summarization instruction (the
 *              SDK ships example/sample_text.txt, ~1K tokens, for this).
 *   --kv       demo the pause / saveKVcache / resume-from-file round trip.
 *   --max-seq  override the model directory's max_seq_len (context/KV
 *              capacity); --init-seq overrides the planned prefill size.
 *              0/omitted = model defaults (SFlareConfig contract).
 */

#include "SFlareApi.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {

/** Streaming sink: the API delivers UTF-8 deltas as they decode. Return 0 to
 *  keep generating; any nonzero return cancels the generation. */
int on_delta(const char *delta, void * /*user_data*/) {
  // [SFLARE_PHASE_TRACE] time-to-first-token marker.
  static bool first = true;
  if (first) {
    first = false;
    if (std::getenv("SFLARE_PHASE_TRACE"))
      std::fprintf(stderr, "[phase-example] first token delivered\n");
  }
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

#if defined(_WIN32)
/** Exit-tail keeper (opt out: SFLARE_EXIT_KEEPER=0). Spawn the bundled
 *  helper in --watch mode right after the model loads: it pings the GPU
 *  once per 100ms while this process lives (negligible contention), then
 *  goes continuous for ~3s after this process dies — so the deferred
 *  kernel-side teardown (see the note at the end of main) runs at awake
 *  speed: measured prompt-return 2.4s -> ~0.4s on Xe3. It must be RUNNING
 *  before death (a spawn at exit is too late to matter), which is why it
 *  is launched here. Silently skipped if the exe isn't next to ours. */
void spawn_exit_keeper() {
  const char *ek = std::getenv("SFLARE_EXIT_KEEPER");
  if (ek && std::strcmp(ek, "0") == 0)
    return;
  char path[MAX_PATH] = {0};
  DWORD n = GetModuleFileNameA(NULL, path, MAX_PATH);
  if (n == 0 || n >= MAX_PATH)
    return;
  char *slash = std::strrchr(path, '\\');
  if (!slash ||
      (slash - path) + std::strlen("\\sflare_exit_keeper.exe") >= MAX_PATH)
    return;
  std::strcpy(slash + 1, "sflare_exit_keeper.exe");
  STARTUPINFOA si = {sizeof(si)};
  PROCESS_INFORMATION pi = {0};
  char cmdline[MAX_PATH + 48];
  std::snprintf(cmdline, sizeof(cmdline), "\"%s\" --watch %lu 3000 100", path,
                (unsigned long)GetCurrentProcessId());
  if (CreateProcessA(path, cmdline, NULL, NULL, FALSE,
                     CREATE_NO_WINDOW | DETACHED_PROCESS, NULL, NULL, &si,
                     &pi)) {
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
  }
}

/** The API takes UTF-8, but Windows hands main() its argv in the ANSI code
 *  page (949/1252/...), so a non-ASCII literal prompt arrives mojibake and
 *  the tokenizer sees garbage. Re-read the command line as UTF-16 and
 *  convert to UTF-8. (A @file prompt is unaffected -- the file is read as
 *  bytes.) Console OUTPUT likewise needs a UTF-8 code page. */
std::vector<std::string> utf8_args() {
  SetConsoleOutputCP(CP_UTF8);
  std::vector<std::string> out;
  int n = 0;
  LPWSTR *wargv = CommandLineToArgvW(GetCommandLineW(), &n);
  if (wargv == nullptr)
    return out;
  for (int i = 0; i < n; ++i) {
    const int len = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0,
                                        nullptr, nullptr);
    std::string s(len > 0 ? len - 1 : 0, '\0');
    if (len > 1)
      WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, &s[0], len, nullptr,
                          nullptr);
    out.push_back(std::move(s));
  }
  LocalFree(wargv);
  return out;
}
#endif

} // namespace

int main(int argc, char *argv[]) {
#if defined(_WIN32)
  // [SFLARE_PHASE_TRACE] how long the DLL constellation + static inits took
  // before main() (import-linked nntrainer.dll & 14 layer DLLs attach first).
  if (std::getenv("SFLARE_PHASE_TRACE")) {
    FILETIME c, e, k, u, now;
    if (GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) {
      GetSystemTimeAsFileTime(&now);
      ULARGE_INTEGER a, b;
      a.LowPart = c.dwLowDateTime;
      a.HighPart = c.dwHighDateTime;
      b.LowPart = now.dwLowDateTime;
      b.HighPart = now.dwHighDateTime;
      std::fprintf(stderr,
                   "[phase-example] main() entered %.0f ms after process "
                   "creation (DLL loads + static init)\n",
                   (b.QuadPart - a.QuadPart) / 10000.0);
    }
  }
  const std::vector<std::string> wargs = utf8_args();
  std::vector<char *> argv_utf8;
  if (!wargs.empty()) {
    for (const auto &a : wargs)
      argv_utf8.push_back(const_cast<char *>(a.c_str()));
    argc = static_cast<int>(argv_utf8.size());
    argv = argv_utf8.data();
  }
#endif
  // Token deltas must reach the console in real time even when stdout is a
  // pipe (bat/ps1 launcher chains) -- unbuffer it up front.
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <model_dir> [cpu|gpu|intel|adreno|cuda] "
                 "[\"prompt\"] [--kv]\n",
                 argv[0]);
    return 1;
  }
  const std::string model_dir = argv[1];
  const std::string backend = argc >= 3 ? argv[2] : "intel";
  std::string prompt =
    argc >= 4 ? argv[3] : "What is the capital of South Korea?";
  bool kv_demo = false;
  unsigned int max_seq = 0, init_seq = 0;
  for (int i = 4; i < argc; ++i) {
    if (std::strcmp(argv[i], "--kv") == 0)
      kv_demo = true;
    else if (std::strcmp(argv[i], "--max-seq") == 0 && i + 1 < argc)
      max_seq = static_cast<unsigned int>(std::atoi(argv[++i]));
    else if (std::strcmp(argv[i], "--init-seq") == 0 && i + 1 < argc)
      init_seq = static_cast<unsigned int>(std::atoi(argv[++i]));
  }

  // @file: summarization demo -- wrap the file's content in a 3-sentence
  // summary instruction (a ~1K-token file makes a realistic prefill).
  bool summarize = false;
  if (!prompt.empty() && prompt[0] == '@') {
    std::ifstream f(prompt.substr(1), std::ios::binary);
    if (!f) {
      std::fprintf(stderr, "[example] cannot open text file: %s\n",
                   prompt.c_str() + 1);
      return 1;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    // Instruction AFTER the passage: with it in front, models tend to slide
    // into continuing the passage instead of summarizing it (measured on
    // gauss4: verbatim continuation with a leading instruction).
    prompt = ss.str() +
             "\n\nSummarize the above text in exactly three sentences.";
    summarize = true;
  }

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
  config.max_seq_len = max_seq;   // 0 = model-directory default
  config.init_seq_len = init_seq; // 0 = model-directory default
  if (!ok(ctx->setSFlareOptions(config), "setSFlareOptions"))
    return 1;

  // 2) Load. This selects the engine (NNTR_ENGINE) and applies the
  //    per-backend env bundle with overwrite=0 -- anything you exported
  //    beforehand wins over the bundle.
  if (!ok(ctx->loadSFlareLLMModel(backend_of(backend)), "loadSFlareLLMModel"))
    return 1;
#if defined(_WIN32)
  spawn_exit_keeper();
#endif

  // Debug aid: what bytes actually reached us (UTF-8 expected on every
  // platform -- see utf8_args()).
  if (std::getenv("SFLARE_EXAMPLE_DEBUG")) {
    std::fprintf(stderr, "[example] prompt %zu bytes:", prompt.size());
    for (size_t i = 0; i < prompt.size() && i < 24; ++i)
      std::fprintf(stderr, " %02x",
                   static_cast<unsigned char>(prompt[i]));
    std::fprintf(stderr, "\n");
  }

  // 3) Streaming generation. GenParams rides both execute overloads;
  //    apply_chat_template=false would feed the prompt raw.
  SFlareApi::GenParams params;
  params.max_new_tokens = summarize ? 128 : 64; // 3 sentences need headroom
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
  //    text appended. loadKVcache() arms the saved cache for the NEXT
  //    execute call, so the resume can use the STREAMING overload too --
  //    tokens appear live, same as the first generation.
  if (kv_demo) {
    std::printf("--- kv save / resume ---\n");
    unsigned int position = 0;
    if (!ok(ctx->pauseSFlareLLM(position), "pauseSFlareLLM"))
      return 1;
    const char *kv_path = "sflare_example_kv.bin";
    if (!ok(ctx->saveKVcache(kv_path), "saveKVcache"))
      return 1;
    std::printf("saved kv at token position %u\n", position);

    if (!ok(ctx->loadKVcache(kv_path, position), "loadKVcache"))
      return 1;
    std::printf("resume (streaming): ");
    if (!ok(ctx->executeSFlareLLM(" Tell me more.", on_delta, nullptr,
                                  &params),
            "executeSFlareLLM(resume,streaming)"))
      return 1;
    std::printf("\n");
  }

  if (!ok(SFlareApi::DestroySFlareContext(ctx), "DestroySFlareContext"))
    return 1;
  std::printf("[example_sflare] done\n");
  // Note: on Windows/Intel iGPU the console prompt returns ~0.8-2.4s after
  // this line (~0.4s with the exit keeper above). Two kernel-side costs land
  // after the process's last instruction (every release call above returns
  // in milliseconds):
  // (1) WDDM VidMm's deferred allocation teardown — destroy calls return
  //     immediately by contract (AssumeNotInUse) and the actual GPU-VA
  //     unmap/unpin of the model pages is force-drained at process death
  //     (~0.6s per 2GiB when the GPU is active);
  // (2) a fixed ~1.5-1.9s device-teardown penalty that latches once the GPU
  //     has accumulated a few seconds of idle while the process held a CL
  //     context — ANY real workload crosses it, an empty hello-world CL
  //     context reproduces it, and no user-mode action (frees, pre-exit GPU
  //     bursts, driver debug keys) avoids or resets it.
  // Both run below the user-mode driver (dxgkrnl/VidMm + KMD); vendor report
  // filed. Amortize by reusing the process for many requests instead of one
  // process per inference.
  return 0;
}

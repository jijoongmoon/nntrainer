// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_module.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   NVRTC compile + PTX disk-cache + Driver-API module load.
 */

#include "cuda_module.h"
#include "cuda_common.h"
#include "cuda_context_manager.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <system_error>
#include <vector>

namespace nntrainer::cuda {

std::size_t Module::GetKernelHash(const std::string &code,
                                  const std::string &options) {
  return std::hash<std::string>{}(code + std::string("\x01") + options);
}

static std::string cacheDir() {
  if (const char *e = getenv("NNTR_CUDA_CACHE"))
    return e;
  const char *home = getenv("HOME");
  return (home ? std::string(home) : std::string("/tmp")) +
         "/.cache/nntrainer_cuda";
}

/// best-effort recursive mkdir (mkdir -p semantics)
static void makeDirs(const std::string &path) {
  // std::filesystem::create_directories creates every missing parent (the
  // recursive case the hand-rolled loop handled) and is portable. Best-effort:
  // swallow errors via the error_code overload (dir may already exist or be
  // uncreatable) — matches the old ignore-return behaviour.
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
}

static bool readFile(const std::string &p, std::string &out) {
  std::ifstream f(p, std::ios::binary);
  if (!f)
    return false;
  std::stringstream ss;
  ss << f.rdbuf();
  out = ss.str();
  return true;
}

static void writeFile(const std::string &p, const std::string &data) {
  std::ofstream f(p, std::ios::binary);
  if (f)
    f.write(data.data(), (std::streamsize)data.size());
}

bool Module::compileWithNVRTC(const std::string &source,
                              const std::string &options, std::string &ptx_out,
                              const std::string &log_tag) {
  nvrtcProgram prog = nullptr;
  if (!nvrtcCheck(nvrtcCreateProgram(&prog, source.c_str(),
                                     (log_tag + ".cu").c_str(), 0, nullptr,
                                     nullptr),
                  "nvrtcCreateProgram"))
    return false;

  std::string arch = "--gpu-architecture=" + ContextManager::Global().GetComputeArch();
  std::vector<std::string> extra;
  {
    std::stringstream ss(options);
    std::string tok;
    while (ss >> tok)
      extra.push_back(tok);
  }
  std::vector<const char *> opts;
  opts.push_back(arch.c_str());
  for (const auto &s : extra)
    opts.push_back(s.c_str());

  nvrtcResult r = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());

  size_t logsz = 0;
  nvrtcGetProgramLogSize(prog, &logsz);
  if (logsz > 1) {
    std::string log(logsz, '\0');
    nvrtcGetProgramLog(prog, &log[0]);
    ml_logd("[NVRTC] %s log:\n%s", log_tag.c_str(), log.c_str());
  }
  if (r != NVRTC_SUCCESS) {
    ml_loge("[NVRTC] compile failed for %s", log_tag.c_str());
    nvrtcDestroyProgram(&prog);
    return false;
  }

  size_t ptxsz = 0;
  if (!nvrtcCheck(nvrtcGetPTXSize(prog, &ptxsz), "nvrtcGetPTXSize")) {
    nvrtcDestroyProgram(&prog);
    return false;
  }
  ptx_out.resize(ptxsz);
  nvrtcGetPTX(prog, &ptx_out[0]);
  nvrtcDestroyProgram(&prog);
  return true;
}

bool Module::CreateModuleFromSource(const std::string &source,
                                    const std::string &name_for_log,
                                    const std::string &options) {
  ContextManager::Global().EnsureCurrent();

  const std::string dir = cacheDir();
  std::string sig = ContextManager::Global().GetDeviceSignature();
  for (auto &c : sig)
    if (c == '|' || c == ' ' || c == '/')
      c = '_';
  const std::string key = dir + "/" + sig + "_" +
                          std::to_string(GetKernelHash(source, options)) + ".ptx";

  std::string ptx;
  bool have_cache = readFile(key, ptx) && !ptx.empty();
  if (!have_cache) {
    if (!compileWithNVRTC(source, options, ptx, name_for_log))
      return false;
    makeDirs(dir);
    writeFile(key, ptx);
  }

  CUresult lr = cuModuleLoadData(&module_, ptx.c_str());
  if (lr != CUDA_SUCCESS && have_cache) {
    // cached PTX may be stale (driver update); recompile once and retry.
    ml_logw("[CUDA] cached PTX load failed; recompiling %s", name_for_log.c_str());
    if (!compileWithNVRTC(source, options, ptx, name_for_log))
      return false;
    writeFile(key, ptx);
    lr = cuModuleLoadData(&module_, ptx.c_str());
  }
  return cuCheck(lr, "cuModuleLoadData");
}

Module::~Module() {
  if (module_) {
    cuModuleUnload(module_);
    module_ = nullptr;
  }
}

} // namespace nntrainer::cuda

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 j2z0.lee <j2z0.lee@ax.samsung.com>
 *
 * @file   htp_context.cpp
 * @date   19 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author j2z0.lee <j2z0.lee@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP compute Context implementation.
 */

#ifdef ENABLE_HEXKL

#include <htp_context.h>

#include <app_context.h>
#include <compute_ops.h>
#include <context_data.h>
#include <htp_backend.h>
#include <htp_mem_allocator.h>

namespace nntrainer {

std::unique_ptr<nntrainer::Layer>
HtpContext::createLayerObject(const std::string &type,
                              const std::vector<std::string> &properties) {
  // HTP has no custom layers; reuse the CPU app context's factories.
  return AppContext::Global().createLayerObject(type, properties);
}

std::unique_ptr<nntrainer::Layer>
HtpContext::createLayerObject(const int int_key,
                              const std::vector<std::string> &properties) {
  return AppContext::Global().createLayerObject(int_key, properties);
}

void HtpContext::initialize() noexcept {
  // Ensure the CPU backend ops table exists — it is both the fallback
  // and the base class of HtpComputeOps.
  init_backend();

  auto cd = getContextData();
  if (!cd)
    return;

  // Wrap the NPU ops installation in try/catch so that a late exception from
  // HtpBackend::global() or get_htp_ops() (e.g., missing skel, driver error)
  // degrades to CPU rather than propagating out of this noexcept function
  // and hitting std::terminate.  Pattern mirrors ClContext::initialize().
  try {
    // NPU up -> HTP ops (shgemm on NPU, rest delegated to CPU via
    // inheritance). NPU down -> plain CPU ops, so an engine="htp" model
    // still runs, transparently, on the CPU.
    if (HtpBackend::global().enabled()) {
      cd->setComputeOps(get_htp_ops());
      cd->setMemAllocator(std::make_shared<HtpMemAllocator>());
    } else
      cd->setComputeOps(get_cpu_ops());
  } catch (...) {
    // HtpBackend::global() or get_htp_ops() threw — degrade to CPU.
    cd->setComputeOps(get_cpu_ops());
  }
}

} // namespace nntrainer

#endif // ENABLE_HEXKL

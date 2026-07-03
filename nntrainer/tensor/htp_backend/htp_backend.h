// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_backend.h
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP (Hexagon Tensor Processor) backend lifecycle.
 *
 * Wraps the Qualcomm HexKL CPU Macro API (sdkl.h / libsdkl.so) NPU
 * lifecycle as a process-wide singleton. Initialization is attempted
 * once; if it fails (no device, missing skel, etc.) the backend is
 * left DISABLED and every HtpComputeOps::supports_*() returns false so
 * callers transparently fall back to the CPU path.
 *
 * Compiled only when ENABLE_HEXKL is defined (meson: -Denable-htp=true).
 */

#ifndef __HTP_BACKEND_H__
#define __HTP_BACKEND_H__
#ifdef __cplusplus
#ifdef ENABLE_HEXKL

namespace nntrainer {

/**
 * @class HtpBackend
 * @brief Process-wide owner of the HexKL NPU session (init/finalize).
 */
class HtpBackend {
public:
  /**
   * @brief Access the process-wide singleton. The first call attempts
   *        sdkl_npu_initialize() exactly once (thread-safe).
   */
  static HtpBackend &global();

  /**
   * @brief Whether the NPU was successfully initialized and is usable.
   *        When false, all HTP ops must defer to the CPU fallback.
   */
  bool enabled() const { return enabled_; }

  /**
   * @brief Target CDSP domain id the session was initialized on.
   */
  int domain() const { return domain_; }

  ~HtpBackend();

  HtpBackend(const HtpBackend &) = delete;
  HtpBackend &operator=(const HtpBackend &) = delete;

private:
  HtpBackend();

  bool enabled_ = false;
  int domain_ = 0; ///< CDSP_DOMAIN_ID (resolved in the .cpp)
};

} // namespace nntrainer

#endif // ENABLE_HEXKL
#endif // __cplusplus
#endif // __HTP_BACKEND_H__

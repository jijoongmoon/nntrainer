// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    env_compat.h
 * @date    09 Jul 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   POSIX setenv() shim for MSVC. The GPU contexts apply their
 *          HW-optimal env defaults with setenv(name, value, overwrite=0)
 *          ("explicit env always wins"); MSVC only has _putenv_s, which
 *          unconditionally overwrites, so the overwrite=0 semantics are
 *          reproduced with a getenv() check. Include this in any TU that
 *          calls setenv() and must also build on Windows. No-op elsewhere.
 */

#ifndef __NNTR_ENV_COMPAT_H__
#define __NNTR_ENV_COMPAT_H__

#if defined(_WIN32)
#include <cstdlib>

static inline int setenv(const char *name, const char *value, int overwrite) {
  if (!overwrite && std::getenv(name) != nullptr)
    return 0;
  return _putenv_s(name, value);
}
#endif

#endif // __NNTR_ENV_COMPAT_H__

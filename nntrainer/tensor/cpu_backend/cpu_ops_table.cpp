// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cpu_ops_table.cpp
 * @date   04 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Unified CPU backend ComputeOps subclass.
 *
 * Single concrete ComputeOps subclass for ALL CPU targets (ARM /
 * x86 / fallback). The nntrainer::sgemm etc. functions are arch-
 * specialized — each arch_compute_backend.cpp defines its own body
 * — so a single forwarding wrapper is enough; build-time arch
 * dispatch picks the right symbol at link time.
 */

#include "cpu_ops_table.h"

namespace nntrainer {

ComputeOps *get_cpu_ops() {
  static CpuComputeOps instance;
  return &instance;
}

} // namespace nntrainer

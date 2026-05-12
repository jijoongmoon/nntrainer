// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 nntrainer authors
 *
 * @file    opencl_profiler.h
 * @date    12 May 2026
 * @brief   Per-kernel GPU timing collector backed by clGetEventProfilingInfo.
 *          Active only when the PROFILE macro is defined (-Denable-profile=true).
 * @see     https://github.com/nntrainer/nntrainer
 * @bug     No known bugs except for NYI items
 */

#ifndef __OPENCL_PROFILER_H__
#define __OPENCL_PROFILER_H__

#ifdef PROFILE

#include <cstdint>
#include <iosfwd>
#include <mutex>
#include <string>
#include <unordered_map>

#include "CL/cl.h"
#include "singleton.h"

namespace nntrainer::opencl {

/**
 * @brief Per-kernel cumulative GPU timing statistics.
 *        All times are nanoseconds reported by the OpenCL driver.
 */
struct KernelTimingStats {
  uint64_t count = 0;        /**< number of dispatches */
  uint64_t queued_ns = 0;    /**< SUBMIT - QUEUED (host queue wait) */
  uint64_t submit_ns = 0;    /**< START  - SUBMIT (device queue wait) */
  uint64_t exec_ns = 0;      /**< END    - START  (actual GPU compute) */
  uint64_t exec_min_ns = ~0ull; /**< minimum per-call exec time */
  uint64_t exec_max_ns = 0;     /**< maximum per-call exec time */
};

/**
 * @brief Singleton that aggregates per-kernel GPU timings.
 *        Only present when nntrainer is built with PROFILE defined.
 */
class OpenCLProfiler : public Singleton<OpenCLProfiler> {
public:
  /**
   * @brief Wait for @a evt to finish then accumulate its profiling info
   *        under @a kernel_name. Releases @a evt.
   *
   * @param kernel_name OpenCL kernel function name (CL_KERNEL_FUNCTION_NAME)
   * @param evt cl_event produced by clEnqueueNDRangeKernel. Must be non-null.
   */
  void record(const std::string &kernel_name, cl_event evt);

  /**
   * @brief Clear all collected statistics.
   */
  void reset();

  /**
   * @brief Print a human-readable summary, sorted by total exec_ns descending.
   *
   * @param out output stream
   */
  void report(std::ostream &out) const;

  /**
   * @brief Whether any sample has been collected.
   */
  bool empty() const;

private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, KernelTimingStats> stats_;
};

} // namespace nntrainer::opencl

#endif // PROFILE
#endif // __OPENCL_PROFILER_H__

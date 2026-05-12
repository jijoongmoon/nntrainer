// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 nntrainer authors
 *
 * @file    opencl_profiler.cpp
 * @date    12 May 2026
 * @brief   Per-kernel GPU timing collector implementation.
 * @see     https://github.com/nntrainer/nntrainer
 * @bug     No known bugs except for NYI items
 */

#ifdef PROFILE

#include "opencl_profiler.h"

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <vector>

#include "opencl_loader.h"

#include <nntrainer_log.h>

namespace nntrainer::opencl {

namespace {

bool query(cl_event evt, cl_profiling_info name, cl_ulong &out) {
  cl_int err =
    clGetEventProfilingInfo(evt, name, sizeof(cl_ulong), &out, nullptr);
  if (err != CL_SUCCESS) {
    ml_loge("clGetEventProfilingInfo failed (param=0x%x), code=%d (%s)",
            (unsigned)name, err, OpenCLErrorCodeToString(err));
    return false;
  }
  return true;
}

} // namespace

void OpenCLProfiler::record(const std::string &kernel_name, cl_event evt) {
  if (evt == nullptr)
    return;

  // Profiling info is only well-defined after the command completes.
  // The PROFILE build accepts this serialization cost for measurement fidelity.
  cl_int werr = clWaitForEvents(1, &evt);
  if (werr != CL_SUCCESS) {
    ml_loge("clWaitForEvents in OpenCLProfiler::record failed, code=%d (%s)",
            werr, OpenCLErrorCodeToString(werr));
    clReleaseEvent(evt);
    return;
  }

  cl_ulong t_queued = 0, t_submit = 0, t_start = 0, t_end = 0;
  bool ok = query(evt, CL_PROFILING_COMMAND_QUEUED, t_queued) &&
            query(evt, CL_PROFILING_COMMAND_SUBMIT, t_submit) &&
            query(evt, CL_PROFILING_COMMAND_START, t_start) &&
            query(evt, CL_PROFILING_COMMAND_END, t_end);
  clReleaseEvent(evt);
  if (!ok)
    return;

  const uint64_t queued_ns = (t_submit > t_queued) ? (t_submit - t_queued) : 0;
  const uint64_t submit_ns = (t_start > t_submit) ? (t_start - t_submit) : 0;
  const uint64_t exec_ns = (t_end > t_start) ? (t_end - t_start) : 0;

  std::lock_guard<std::mutex> g(mutex_);
  auto &s = stats_[kernel_name];
  s.count += 1;
  s.queued_ns += queued_ns;
  s.submit_ns += submit_ns;
  s.exec_ns += exec_ns;
  s.exec_min_ns = std::min(s.exec_min_ns, exec_ns);
  s.exec_max_ns = std::max(s.exec_max_ns, exec_ns);
}

void OpenCLProfiler::reset() {
  std::lock_guard<std::mutex> g(mutex_);
  stats_.clear();
}

bool OpenCLProfiler::empty() const {
  std::lock_guard<std::mutex> g(mutex_);
  return stats_.empty();
}

void OpenCLProfiler::report(std::ostream &out) const {
  std::lock_guard<std::mutex> g(mutex_);
  if (stats_.empty()) {
    out << "[OpenCLProfiler] no samples\n";
    return;
  }

  std::vector<std::pair<std::string, KernelTimingStats>> rows(stats_.begin(),
                                                              stats_.end());
  std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) {
    return a.second.exec_ns > b.second.exec_ns;
  });

  uint64_t grand_exec = 0;
  for (const auto &r : rows)
    grand_exec += r.second.exec_ns;

  out << "\n=== OpenCL GPU Profile ===\n";
  out << std::left << std::setw(48) << "kernel" << std::right << std::setw(8)
      << "calls" << std::setw(14) << "exec_total_us" << std::setw(12)
      << "avg_us" << std::setw(12) << "min_us" << std::setw(12) << "max_us"
      << std::setw(14) << "queued_us" << std::setw(14) << "submit_us"
      << std::setw(8) << "%exec"
      << "\n";
  out << std::string(48 + 8 + 14 + 12 + 12 + 12 + 14 + 14 + 8, '-') << "\n";

  for (const auto &r : rows) {
    const auto &s = r.second;
    const double avg_us =
      s.count ? (double)s.exec_ns / s.count / 1000.0 : 0.0;
    const double pct =
      grand_exec ? 100.0 * (double)s.exec_ns / (double)grand_exec : 0.0;

    out << std::left << std::setw(48) << r.first.substr(0, 47) << std::right
        << std::setw(8) << s.count << std::setw(14) << (s.exec_ns / 1000)
        << std::fixed << std::setprecision(2) << std::setw(12) << avg_us
        << std::setw(12) << (s.exec_min_ns / 1000) << std::setw(12)
        << (s.exec_max_ns / 1000) << std::setw(14) << (s.queued_ns / 1000)
        << std::setw(14) << (s.submit_ns / 1000) << std::setw(7) << pct << "%"
        << std::defaultfloat << "\n";
  }

  out << std::string(48 + 8 + 14 + 12 + 12 + 12 + 14 + 14 + 8, '-') << "\n";
  out << std::left << std::setw(48) << "TOTAL" << std::right << std::setw(8)
      << "" << std::setw(14) << (grand_exec / 1000) << "\n";
}

} // namespace nntrainer::opencl

#endif // PROFILE

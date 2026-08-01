// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    performance_metrics.h
 * @date    24 Mar 2026
 * @brief   Performance metrics definitions shared between models and API layers
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __CAUSAL_LM_PERFORMANCE_METRICS_H__
#define __CAUSAL_LM_PERFORMANCE_METRICS_H__

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Performance Metrics
 */
typedef struct {
  unsigned int prefill_tokens;
  double prefill_duration_ms;
  unsigned int generation_tokens;
  double generation_duration_ms;
  double total_duration_ms;
  double initialization_duration_ms;
  size_t peak_memory_kb;
} TransformerPerformanceMetrics;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <psapi.h>
#else
#include <sys/resource.h>
#endif

/**
 * @brief Get peak memory usage in KB
 */
inline size_t getPeakMemoryKb() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    return (size_t)(pmc.PeakWorkingSetSize / 1024);
  }
  return 0;
#else
  struct rusage rusage;
  if (getrusage(RUSAGE_SELF, &rusage) == 0) {
    return (size_t)(rusage.ru_maxrss);
  }
  return 0;
#endif
}

/**
 * @brief Peak private commit (pagefile-backed) in KB, or 0 if unavailable.
 * @note Complements getPeakMemoryKb() (peak working set). Working set counts
 *  physically-resident pages INCLUDING shareable file-backed ones (mmap'd
 *  weights, DLLs) and is trimmed by the OS under pressure; peak commit counts
 *  only this process's private, committed VA and is what the system commit
 *  limit charges. For memory-footprint questions (capacity sizing, leak
 *  detection, KV-quantization savings) commit is the honest number -- large
 *  coarse GPU-SVM/UVM slabs are only fractionally charged to working set, so
 *  a KV-cache halving can be invisible in WS yet visible in commit. On
 *  non-Windows peak commit is not exposed via getrusage -> returns 0.
 */
inline size_t getPeakCommitKb() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS_EX pmcx;
  pmcx.cb = sizeof(pmcx);
  if (GetProcessMemoryInfo(GetCurrentProcess(),
                           reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&pmcx),
                           sizeof(pmcx))) {
    return (size_t)(pmcx.PeakPagefileUsage / 1024);
  }
  return 0;
#else
  return 0; // getrusage exposes peak RSS only, not peak commit
#endif
}

#endif // __cplusplus

#endif // __CAUSAL_LM_PERFORMANCE_METRICS_H__

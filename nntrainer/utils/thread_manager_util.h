// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jaemin Shin <jaemin2.shin@samsung.com>
 *
 * @file   thread_manager_util.h
 * @date   23 April 2026
 * @brief  Utils for unified thread manager
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jaemin Shin <jaemin2.shin@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_THREAD_MANAGER_UTIL_H__
#define __NNTRAINER_THREAD_MANAGER_UTIL_H__

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace nntrainer {

/**
 * @brief read uint32_t value from a given file
 */
uint32_t readUInt(const std::string &path);

/**
 * @brief read string value from a given file
 */
std::string readStr(const std::string &path);

/**
 * @brief parse string into following form
 *   * "0-3,8" -> {0,1,2,3,8}
 */
std::vector<uint32_t> parseCpuList(const std::string &s);

/**
 * @brief get count of physical core considering SMT
 */
uint32_t getPhysicalCoreCount();

/**
 * @brief default number of compute threads (main thread included) when
 * neither NNTR_NUM_THREADS nor the nntr-num-threads build option names one.
 *
 * x86: the physical cores this process may run on (affinity mask, SMT
 * siblings counted once), capped at 16. The CPU kernels are compute-bound,
 * so throughput follows the core count; past 16 the shared memory bandwidth
 * and the pool's spin-wait stop paying back.
 * Others: std::thread::hardware_concurrency() / 2, as before.
 */
uint32_t getDefaultComputeThreadCount();

/**
 * @brief return list of cores in performance order
 */
std::vector<uint32_t> getCoresByPerformance();

/**
 * @brief Pin the CALLING thread to a specific CPU core.
 */
bool pinSelfToCore(uint32_t core_id);

} // namespace nntrainer

#endif // __NNTRAINER_THREAD_MANAGER_UTIL_H__

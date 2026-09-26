/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file nntrainer_logger.cpp
 * @date 02 April 2020
 * @brief NNTrainer Logger
 *        This allows to logging nntrainer logs.
 * @see	https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <nntrainer_logger.h>
#include <sstream>
#include <stdarg.h>
#include <stdexcept>
#include <system_error>
#include <util_func.h>
#include <vector>

namespace nntrainer {

namespace {

/**
 * @brief     Retention defaults for the per-user log directory: at most this
 *            many rotated files, and at most this many MiB total. See
 *            docs/ENV_FLAGS.md (NNTR_LOG_KEEP / NNTR_LOG_MAX_MB).
 */
constexpr std::uintmax_t default_log_keep_count = 20;
constexpr std::uintmax_t default_log_max_mb = 64;

/**
 * @brief     Parse a non-negative integer out of @a v (as read from an
 *            environment variable), falling back to @a fallback when @a v is
 *            null, empty, or not a valid non-negative integer.
 */
std::uintmax_t parseUintOr(const char *v, std::uintmax_t fallback) {
  if (v == nullptr || *v == '\0')
    return fallback;
  try {
    size_t pos = 0;
    long long parsed = std::stoll(v, &pos);
    if (pos != std::strlen(v) || parsed < 0)
      return fallback;
    return static_cast<std::uintmax_t>(parsed);
  } catch (...) {
    return fallback;
  }
}

} // namespace

/**
 * @brief     logfile name
 */
const char *const Logger::logfile_name = "log_nntrainer_";
const char *const Logger::logfile_subdir = "logs";
/**
 * @brief     instance for single logger
 */
Logger *Logger::ainstance = nullptr;

/**
 * @brief     mutex for lock
 */
std::mutex Logger::smutex;

Logger &Logger::instance() {
  static Cleanup cleanup;

  std::lock_guard<std::mutex> guard(smutex);
  if (ainstance == nullptr)
    ainstance = new Logger();
  return *ainstance;
}

Logger::Cleanup::~Cleanup() {
  std::lock_guard<std::mutex> guard(Logger::smutex);
  delete Logger::ainstance;
  Logger::ainstance = nullptr;
}

Logger::~Logger() {
  try {
    outputstream.close();
  } catch (...) {
    std::cerr << "Error closing the log file\n";
  }
}

std::string Logger::resolveLogDir() {
  return resolveUserDataDir("NNTR_LOG_DIR", logfile_subdir);
}

bool Logger::openLogFile(const std::string &dir, const std::string &file_name,
                         std::ofstream &out) noexcept {
  try {
    if (dir.empty())
      return false;
    // The error_code overload: the throwing one raises
    // std::filesystem::filesystem_error for a directory that cannot be
    // created, and the logger is first constructed inside whatever call
    // logged first -- model load, typically -- so that exception used to
    // take the whole load down.
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec)
      return false;
    out.open((std::filesystem::path(dir) / file_name).string(),
             std::ios_base::app);
    if (!out.is_open() || !out.good()) {
      out.close();
      return false;
    }
    return true;
  } catch (...) {
    return false;
  }
}

void Logger::pruneLogDir(const std::string &dir,
                         const std::string &keep_file) noexcept {
  try {
    if (dir.empty())
      return;

    const std::uintmax_t keep_count =
      parseUintOr(std::getenv("NNTR_LOG_KEEP"), default_log_keep_count);
    const std::uintmax_t max_mb =
      parseUintOr(std::getenv("NNTR_LOG_MAX_MB"), default_log_max_mb);
    if (keep_count == 0 && max_mb == 0)
      return; // both limits disabled

    const std::uintmax_t max_bytes = max_mb * 1024ull * 1024ull;
    const std::string prefix = logfile_name; // "log_nntrainer_"
    const std::string suffix = ".out";

    std::filesystem::path keep_path;
    if (!keep_file.empty())
      keep_path = std::filesystem::path(keep_file);

    struct Entry {
      std::filesystem::path path;
      std::filesystem::file_time_type mtime;
      std::uintmax_t size;
    };
    std::vector<Entry> entries;

    std::error_code ec;
    auto it = std::filesystem::directory_iterator(dir, ec);
    if (ec)
      return;
    for (; it != std::filesystem::directory_iterator(); it.increment(ec)) {
      if (ec)
        break;
      const std::filesystem::path p = it->path();
      const std::string fname = p.filename().string();
      if (fname.rfind(prefix, 0) != 0 || fname.size() < suffix.size() ||
          fname.compare(fname.size() - suffix.size(), suffix.size(), suffix) !=
            0)
        continue;
      if (!keep_path.empty()) {
        std::error_code eq_ec;
        if (std::filesystem::equivalent(p, keep_path, eq_ec) && !eq_ec)
          continue;
      }
      std::error_code type_ec;
      if (!it->is_regular_file(type_ec) || type_ec)
        continue;
      std::error_code size_ec, time_ec;
      const std::uintmax_t size = std::filesystem::file_size(p, size_ec);
      const auto mtime = std::filesystem::last_write_time(p, time_ec);
      if (size_ec || time_ec)
        continue;
      entries.push_back(Entry{p, mtime, size});
    }

    std::sort(entries.begin(), entries.end(),
              [](const Entry &a, const Entry &b) { return a.mtime < b.mtime; });

    // the file just opened for this process is not in `entries` (it was
    // excluded above), but it still counts toward both limits.
    std::uintmax_t count = entries.size() + (keep_path.empty() ? 0 : 1);
    std::uintmax_t size_total = 0;
    for (const auto &e : entries)
      size_total += e.size;

    bool warned = false;
    for (const auto &e : entries) {
      const bool over_count = keep_count != 0 && count > keep_count;
      const bool over_size = max_mb != 0 && size_total > max_bytes;
      if (!over_count && !over_size)
        break;
      std::error_code rm_ec;
      std::filesystem::remove(e.path, rm_ec);
      if (rm_ec) {
        if (!warned) {
          std::cerr << "nntrainer: could not prune old log file "
                    << e.path.string() << ": " << rm_ec.message() << std::endl;
          warned = true;
        }
        continue;
      }
      count -= 1;
      size_total -= e.size;
    }
  } catch (...) {
    // Fail-soft: pruning is housekeeping and must never take model load
    // down with it.
  }
}

Logger::Logger() : ts_type(NNTRAINER_LOG_TIMESTAMP_SEC) {
  struct tm now;
  getLocaltime(&now);
  std::stringstream ss;
  ss << logfile_name << std::dec << (now.tm_year + 1900) << std::setfill('0')
     << std::setw(2) << (now.tm_mon + 1) << std::setfill('0') << std::setw(2)
     << now.tm_mday << std::setfill('0') << std::setw(2) << now.tm_hour
     << std::setfill('0') << std::setw(2) << now.tm_min << std::setfill('0')
     << std::setw(2) << now.tm_sec << ".out";

  // Fail-soft: logging is diagnostics, and a directory that cannot be written
  // must never cost the caller its model load. Without a log file, warnings
  // and errors go to stderr (see log()) and the rest is dropped.
  const std::string dir = resolveLogDir();
  const std::string file_name = ss.str();
  const bool opened = !dir.empty() && openLogFile(dir, file_name, outputstream);
  if (!dir.empty() && !opened) {
    std::cerr << "nntrainer: cannot write log files under " << dir
              << "; file logging disabled, warnings and errors go to stderr"
              << std::endl;
  } else if (opened) {
    // A performance run that spawns many short-lived processes (each one a
    // Logger) otherwise leaves one file per process forever under the
    // per-user directory; keep it bounded the same way the kernel cache is.
    pruneLogDir(dir, (std::filesystem::path(dir) / file_name).string());
  }
}

void Logger::log(const std::string &message,
                 const nntrainer_loglevel loglevel) {
  std::lock_guard<std::mutex> guard(smutex);
  std::stringstream ss;

  switch (loglevel) {
  case NNTRAINER_LOG_INFO:
    ss << "[NNTRAINER INFO  ";
    break;
  case NNTRAINER_LOG_WARN:
    ss << "[NNTRAINER WARN  ";
    break;
  case NNTRAINER_LOG_ERROR:
    ss << "[NNTRAINER ERROR ";
    break;
  case NNTRAINER_LOG_DEBUG:
    ss << "[NNTRAINER DEBUG ";
    break;
  default:
    break;
  }

  if (ts_type == NNTRAINER_LOG_TIMESTAMP_MS) {
    static auto start = std::chrono::system_clock::now().time_since_epoch();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch() - start)
                .count();

    ss << "[ " << ms << " ]";
  } else if (ts_type == NNTRAINER_LOG_TIMESTAMP_SEC) {
    struct tm now;
    getLocaltime(&now);

    ss << std::dec << (now.tm_year + 1900) << '-' << std::setfill('0')
       << std::setw(2) << (now.tm_mon + 1) << '-' << std::setfill('0')
       << std::setw(2) << now.tm_mday << ' ' << std::setfill('0')
       << std::setw(2) << now.tm_hour << ':' << std::setfill('0')
       << std::setw(2) << now.tm_min << ':' << std::setfill('0') << std::setw(2)
       << now.tm_sec << ']';
  }

  if (outputstream.is_open())
    outputstream << ss.str() << " " << message << std::endl;
  else if (loglevel >= NNTRAINER_LOG_WARN)
    std::cerr << ss.str() << " " << message << std::endl;
}

} /* namespace nntrainer */

#ifdef __cplusplus
extern "C" {
#endif
void __nntrainer_log_print(nntrainer_loglevel loglevel,
                           const std::string format_str, ...) {
  int final_n, n = ((int)format_str.size()) * 2;
  std::unique_ptr<char[]> formatted;
  va_list ap;
  while (1) {
    formatted.reset(new char[n]);
    std::strncpy(&formatted[0], format_str.c_str(), format_str.size());
    va_start(ap, format_str);
    final_n = vsnprintf(&formatted[0], n, format_str.c_str(), ap);
    va_end(ap);
    if (final_n < 0 || final_n >= n)
      n += abs(final_n - n + 1);
    else
      break;
  }

  std::string ss = std::string(formatted.get());

#if defined(__LOGGING__)
  nntrainer::Logger::instance().log(ss, loglevel);
#else

#if defined(DEBUG)
  switch (loglevel) {
  case NNTRAINER_LOG_ERROR:
    std::cerr << ss << std::endl;
    break;
  case NNTRAINER_LOG_INFO:
  case NNTRAINER_LOG_WARN:
  case NNTRAINER_LOG_DEBUG:
    std::cout << ss << std::endl;
  default:
    break;
  }
#endif
#endif
}

#ifdef __cplusplus
}
#endif

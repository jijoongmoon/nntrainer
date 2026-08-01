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
 *
 * @file	util_func.cpp
 * @date	08 April 2020
 * @brief	This is collection of math functions
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifdef _WIN32
#define MAX_PATH_LENGTH 1024
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <random>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <acti_func.h>
#include <nntrainer_log.h>
#include <util_func.h>

namespace nntrainer {

static std::uniform_real_distribution<float> dist(-0.5, 0.5);

double sqrtDouble(double x) { return sqrt(x); };

float logFloat(float x) { return log(x + 1.0e-20); }

float exp_util(float x) { return exp(x); }

uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; };

uint32_t align(uint32_t a, uint32_t b) {
  return (a % b == 0) ? a : a - a % b + b;
};

Tensor rotate_180(Tensor in) {
  Tensor output(in.getDim());
  output.setZero();
  for (unsigned int i = 0; i < in.batch(); ++i) {
    for (unsigned int j = 0; j < in.channel(); ++j) {
      for (unsigned int k = 0; k < in.height(); ++k) {
        for (unsigned int l = 0; l < in.width(); ++l) {
          output.setValue(
            i, j, k, l,
            in.getValue(i, j, (in.height() - k - 1), (in.width() - l - 1)));
        }
      }
    }
  }
  return output;
}

bool isFileExist(std::string file_name) {
  std::ifstream infile(file_name);
  return infile.good();
}

template <typename T>
static void checkFile(const T &file, const char *error_msg) {
  if (file.bad() || file.eof() || !file.good() || file.fail()) {
    throw std::runtime_error(error_msg);
  }
}

#if !defined(_WIN32)
namespace {

/**
 * The one staging window a load may register (see LoaderStagingMap). Read
 * lock-free by every load worker from checkedRead below, written twice per
 * load by the parent: once before the workers start, once after they join.
 * `window_base` is the CLAIM -- it is compare-exchanged, so exactly one load
 * owns the window -- and `window_end` is the PUBLICATION point, stored after
 * the claim with release and loaded with acquire. A reader that sees a base
 * without its matching end sees end == nullptr, fails containment, and copies
 * plainly; it never sees one load's base paired with another's end.
 */
std::atomic<const char *> window_base{nullptr};
std::atomic<const char *> window_end{nullptr};
std::atomic<size_t> window_chunk{0};
/** accounting, reported once per load so the release is observable */
std::atomic<size_t> window_released{0};
std::atomic<size_t> window_calls{0};

size_t pageSize() {
  static const size_t p = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
  return p;
}

/**
 * @brief Copy out of the staging window, releasing whole pages behind the
 *        copy cursor.
 *
 * The release range is always [alignUp(src), alignDown(src + copied)), so it
 * is contained in this record's own bytes: a page shared with the neighbouring
 * record -- which another worker may be mid-copy on -- is never dropped, and
 * no page is dropped before its bytes have been consumed. Each page is
 * therefore faulted once and released once.
 */
void stagedCopy(char *dst, const char *src, size_t n, size_t chunk) {
  const size_t page = pageSize();
  uintptr_t released =
    (reinterpret_cast<uintptr_t>(src) + page - 1) & ~(page - 1);
  size_t done = 0;
  size_t freed = 0;
  size_t calls = 0;

  while (done < n) {
    const size_t take = std::min(chunk, n - done);
    std::memcpy(dst + done, src + done, take);
    done += take;

    const uintptr_t cursor =
      reinterpret_cast<uintptr_t>(src + done) & ~(page - 1);
    if (cursor > released) {
      const size_t len = static_cast<size_t>(cursor - released);
      if (::madvise(reinterpret_cast<void *>(released), len, MADV_DONTNEED) ==
          0) {
        freed += len;
        ++calls;
      }
      released = cursor;
    }
  }

  if (calls) {
    window_released.fetch_add(freed, std::memory_order_relaxed);
    window_calls.fetch_add(calls, std::memory_order_relaxed);
  }
}

} // namespace

size_t LoaderStagingMap::budgetBytes() {
  // Same name, same units and same meaning the sampling reaper gave it: how
  // many MiB of source-file staging the load may hold resident. Default is on
  // -- the release costs one madvise per chunk and never re-faults, so there
  // is nothing for a caller to trade away; 0 turns it off for A/B only.
  constexpr size_t DEFAULT_BUDGET_MB = 128;
  const char *e = std::getenv("NNTR_LOAD_REAP_MB");
  const size_t mb =
    e ? static_cast<size_t>(std::strtoul(e, nullptr, 10)) : DEFAULT_BUDGET_MB;
  return mb << 20;
}

LoaderStagingMap::LoaderStagingMap(int fd, size_t length, size_t concurrency,
                                   const char *what) :
  base_(nullptr), length_(length), registered_(false) {
  void *p = ::mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, 0);
  NNTR_THROW_IF((p == MAP_FAILED), std::runtime_error)
    << "mmap failed for " << (what ? what : "model file") << " (" << length
    << " bytes)";
  base_ = static_cast<char *>(p);

  // Warm the mapping with readahead. MADV_RANDOM was actively harmful here:
  // each worker reads its node's tensors as a sequential sub-range, so
  // suppressing readahead made every page fault individually. WILLNEED lets
  // the workers hit warm page cache instead, which is what keeps the faults
  // minor and cheap now that each page is faulted exactly once.
  (void)::posix_madvise(base_, length_, POSIX_MADV_WILLNEED);

  const size_t budget = budgetBytes();
  if (budget == 0)
    return;

  // The bound this class exists to provide: staging residency is at most
  // (workers * chunk). Dividing the budget by the worker count is what makes
  // the bound hold for a loader that fans out one thread per node as well as
  // for one that caps at eight. The floor keeps the madvise call rate sane on
  // the wide fan-out (below it the calls, not the pages, would dominate); it
  // is the only place the bound is loosened, and it loosens it to
  // (workers * 512 KiB), still independent of any record's size.
  constexpr size_t MIN_CHUNK = 512u << 10;
  const size_t chunk =
    std::max(MIN_CHUNK, budget / std::max<size_t>(concurrency, 1));

  // Claim ownership FIRST, publish the window's extent second. Writing
  // window_end before the compare_exchange decided the owner let a load that
  // loses the claim overwrite the extent of the window that won it: the live
  // window became [winner base, loser end), one range spanning two unrelated
  // mappings, and the release then ran outside the mapping it belongs to
  // (measured: "released 664 MiB ... over a 333 MiB mapping" on two concurrent
  // loads of one 333 MiB package). The loser also zeroed the winner's
  // accounting. Only the owner touches any of these now.
  //
  // The ordering is safe for the readers in checkedRead: they take base with
  // acquire and end with acquire, and a reader that catches the gap between
  // the claim and the extent store sees end == nullptr -- the value the
  // previous owner's destructor left -- which fails containment and takes the
  // plain memcpy. It can lose the release for that copy; it can never aim a
  // madvise anywhere.
  const char *expected = nullptr;
  if (window_base.compare_exchange_strong(expected, base_,
                                          std::memory_order_acq_rel,
                                          std::memory_order_relaxed)) {
    window_chunk.store(chunk, std::memory_order_relaxed);
    window_released.store(0, std::memory_order_relaxed);
    window_calls.store(0, std::memory_order_relaxed);
    window_end.store(base_ + length_, std::memory_order_release);
    registered_ = true;
  } else {
    // Another load already owns the window. The mapping is still perfectly
    // usable, it just does not get the per-chunk release; say so rather than
    // silently pointing the window at the wrong file.
    ml_logw("loader staging: a staging window is already registered; %s will "
            "not release its pages until the load ends",
            (what ? what : "this mapping"));
  }
}

LoaderStagingMap::~LoaderStagingMap() {
  if (registered_) {
    const size_t freed = window_released.load(std::memory_order_relaxed);
    const size_t calls = window_calls.load(std::memory_order_relaxed);
    // Retire the extent before releasing the claim, the mirror of the
    // constructor: once base_ is clearable another load may claim it
    // immediately, and clearing end afterwards would wipe the extent that
    // load just published.
    window_end.store(nullptr, std::memory_order_release);
    window_chunk.store(0, std::memory_order_relaxed);
    window_base.store(nullptr, std::memory_order_release);
    ml_logi("loader staging: released %zu MiB in %zu madvise calls over a %zu "
            "MiB mapping",
            freed >> 20, calls, length_ >> 20);
  }
  if (base_ != nullptr) {
    // Releasing before the unmap keeps the last in-flight records from being
    // charged to the process any longer than the copies themselves.
    (void)::madvise(base_, length_, MADV_DONTNEED);
    ::munmap(base_, length_);
    base_ = nullptr;
  }
}
#endif

void checkedRead(std::ifstream &file, char *array, std::streamsize size,
                 const char *error_msg, size_t start_offset,
                 bool read_from_offset) {
  if (read_from_offset) {
    file.seekg(start_offset, std::ios::beg);
    checkFile(file, "failed to move offset");
  }
  file.read(array, size);
  checkFile(file, error_msg);
}

void checkedRead(ReadSource src, char *array, std::streamsize size,
                 const char *error_msg, size_t start_offset,
                 bool read_from_offset) {

  if (auto f = std::get_if<std::ifstream *>(&src)) {
    if (read_from_offset) {
      (*f)->seekg(start_offset, std::ios::beg);
      checkFile(**f, "failed to move offset");
    }
    (*f)->read(static_cast<char *>(array), static_cast<std::streamsize>(size));
    // The stream overload above has always checked this; here it was commented
    // out, and the reason it had to be is that `*f` is an std::ifstream* -- the
    // template would have deduced T = std::ifstream* and failed to compile on
    // .bad(). Dereference once more. Without it a short read (a truncated or
    // mis-offset weight file) leaves the destination holding whatever was in
    // the buffer and the caller none the wiser; istream::read only sets eofbit
    // when it extracted FEWER bytes than asked, so a read that lands exactly on
    // EOF is still accepted.
    checkFile(**f, error_msg);
  } else if (auto p = std::get_if<const char *>(&src)) {
    const char *from = read_from_offset ? (*p) + start_offset : (*p);
    const size_t n = static_cast<size_t>(size);

#if !defined(_WIN32)
    // This is the single mmap -> weight-pool copy of the whole loader, so it
    // is where the source-file staging residency is bounded. If the source
    // lies inside the loader's registered staging window (LoaderStagingMap),
    // copy in chunks and release each finished page behind the cursor.
    const char *base = window_base.load(std::memory_order_acquire);
    const char *end =
      base ? window_end.load(std::memory_order_acquire) : nullptr;
    // Containment is a property of THIS POINTER, not of "a window happens to
    // be registered". Distinct mappings never overlap, so a source that is not
    // this window's -- a caller-owned buffer, or the mapping of a SECOND
    // concurrent NeuralNetwork::load, which lost the compare_exchange in
    // LoaderStagingMap and is documented there to keep running without the
    // release -- lies wholly outside [base, end) and must take the plain
    // memcpy below. Keying the decision on window presence instead made every
    // such read throw unconditionally, on a load worker, which is
    // std::terminate.
    if (base != nullptr && from >= base && from < end) {
      // Inside the window: the release below is now permitted (madvise must
      // never be aimed outside the mapping) and the length IS known, which is
      // the check this branch could not make before. A `const char *` mapping
      // carries no length, so an over-long size or an out-of-range
      // start_offset used to read past the end of the mapping and SIGSEGV on a
      // load worker thread, ~300 tensors away from whichever layer mis-sized
      // itself. Name the read that escaped instead. (The layout tripwire in
      // NeuralNetwork::load still catches the common case earlier and with
      // more context; do not remove it on the strength of this one.)
      if (n > static_cast<size_t>(end - from)) {
        // Log BEFORE throwing. This runs on a load worker, where a throw is
        // std::terminate, and with eight workers failing at once the runtime
        // prints "terminate called recursively" and the what() string is lost
        // -- verified by forcing this path. The log file is the record that
        // survives.
        ml_loge("[util::checkedRead] read of %zu bytes at offset %zu escapes "
                "the model file mapping of %zu bytes: a weight's requested "
                "dtype/size disagrees with the stored record",
                n, static_cast<size_t>(from - base),
                static_cast<size_t>(end - base));
        throw std::runtime_error(
          "[util::checkedRead] read escapes the model file mapping; see the "
          "nntrainer log for the offending offset and size");
      }
      stagedCopy(array, from, n, window_chunk.load(std::memory_order_relaxed));
      return;
    }
#endif

    /// @todo use mmap instead memcpy to reduce peak memory
    // Not the staging window: a caller-owned buffer (the Windows file view, a
    // training-mode read, a second concurrent load's mapping), whose length
    // this branch cannot know and whose pages it must not touch.
    std::memcpy(array, from, n);
  }
}

void checkedWrite(std::ostream &file, const char *array, std::streamsize size,
                  const char *error_msg) {
  file.write(array, size);

  checkFile(file, error_msg);
}

std::string readString(std::ifstream &file, const char *error_msg) {
  std::string str;
  size_t size;

  checkedRead(file, (char *)&size, sizeof(size), error_msg);

  std::streamsize sz = static_cast<std::streamsize>(size);
  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "read string size: " << sz
    << " is too big. It cannot be represented by std::streamsize";

  str.resize(size);
  checkedRead(file, (char *)&str[0], sz, error_msg);

  return str;
}

void writeString(std::ofstream &file, const std::string &str,
                 const char *error_msg) {
  size_t size = str.size();

  checkedWrite(file, (char *)&size, sizeof(size), error_msg);

  std::streamsize sz = static_cast<std::streamsize>(size);
  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "write string size: " << size
    << " is too big. It cannot be represented by std::streamsize";

  checkedWrite(file, (char *)&str[0], sz, error_msg);
}

bool endswith(const std::string &target, const std::string &suffix) {
  if (target.size() < suffix.size()) {
    return false;
  }
  size_t spos = target.size() - suffix.size();
  return target.substr(spos) == suffix;
}

int getKeyValue(const std::string &input_str, std::string &key,
                std::string &value) {
  int status = ML_ERROR_NONE;
  auto input_trimmed = input_str;

  std::vector<std::string> list;
  static const std::regex words_regex("[^\\s=]+");
  input_trimmed.erase(
    std::remove(input_trimmed.begin(), input_trimmed.end(), ' '),
    input_trimmed.end());
  auto words_begin = std::sregex_iterator(input_trimmed.begin(),
                                          input_trimmed.end(), words_regex);
  auto words_end = std::sregex_iterator();
  int nwords = std::distance(words_begin, words_end);

  if (nwords != 2) {
    ml_loge("Error: input string must be 'key = value' format "
            "(e.g.{\"key1=value1\",\"key2=value2\"}), \"%s\" given",
            input_trimmed.c_str());
    return ML_ERROR_INVALID_PARAMETER;
  }

  for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
    list.push_back((*i).str());
  }

  key = list[0];
  value = list[1];

  return status;
}

int getValues(int n_str, std::string str, int *value) {
  int status = ML_ERROR_NONE;
  static const std::regex words_regex("[^\\s.,:;!?]+");
  str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
  auto words_begin = std::sregex_iterator(str.begin(), str.end(), words_regex);
  auto words_end = std::sregex_iterator();

  int num = std::distance(words_begin, words_end);
  if (num != n_str) {
    ml_loge("Number of Data is not match");
    return ML_ERROR_INVALID_PARAMETER;
  }
  int cn = 0;
  for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
    value[cn] = std::stoi((*i).str());
    cn++;
  }
  return status;
}

std::vector<std::string> split(const std::string &s, const std::regex &reg) {
  std::vector<std::string> out;
  const int NUM_SKIP_CHAR = 3;
  char char_to_remove[NUM_SKIP_CHAR] = {' ', '[', ']'};
  std::string str = s;
  for (unsigned int i = 0; i < NUM_SKIP_CHAR; ++i) {
    str.erase(std::remove(str.begin(), str.end(), char_to_remove[i]),
              str.end());
  }

  std::regex_token_iterator<std::string::iterator> end;
  std::regex_token_iterator<std::string::iterator> iter(str.begin(), str.end(),
                                                        reg, -1);

  while (iter != end) {
    out.push_back(*iter);
    ++iter;
  }
  return out;
}

bool istrequal(const std::string &a, const std::string &b) {
  if (a.size() != b.size())
    return false;

  return std::equal(a.begin(), a.end(), b.begin(), [](char a_, char b_) {
    return tolower(a_) == tolower(b_);
  });
}

char *getRealpath(const char *name, char *resolved) {
#ifdef _WIN32
  return _fullpath(resolved, name, MAX_PATH_LENGTH);
#else
  resolved = realpath(name, nullptr);
  return resolved;
#endif
}

tm *getLocaltime(tm *tp) {
  time_t t = time(0);
#ifdef _WIN32
  localtime_s(tp, &t);
  return tp;
#else
  return localtime_r(&t, tp);
#endif
}

std::regex getRegex(const std::string &str) {
  std::regex result;

  try {
    result = std::regex(str);
  } catch (const std::regex_error &e) {
    ml_loge("regex_error caught: %s", e.what());
  }

  return result;
}

void floatToFixedPointAndExponent(float input, int &fixedpoint, int &exponent) {
  exponent = 0;
  // normalize the floating-point number into the form: mantissa * 2^exponent
  float mantissa = std::frexp(input, &exponent);
  // scale mantissa to a fixed-point range to maximize precision
  fixedpoint = static_cast<int>(
    mantissa * static_cast<float>(std::numeric_limits<int>::max()));
}

float fixedPointAndExponentToFloat(int fixedpoint, int exponent) {
  // scale back to the normalized floating-point range
  float mantissa = static_cast<float>(fixedpoint) /
                   static_cast<float>(std::numeric_limits<int>::max());
  // reconstruct the floating-point number
  return std::ldexp(mantissa, exponent);
}

} // namespace nntrainer

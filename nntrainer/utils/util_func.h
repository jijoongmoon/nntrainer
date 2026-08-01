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
 * @file	util_func.h
 * @date	08 April 2020
 * @brief	This is collection of math functions.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __UTIL_FUNC_H__
#define __UTIL_FUNC_H__

#ifdef __cplusplus

#include <cstring>
#include <regex>
#include <sstream>

#include <nntrainer_error.h>
#include <random>
#include <variant>

// /**
//  * @brief     get the seed
//  * @return    seed
//  */
// unsigned int getSeed() { return 0; }

namespace nntrainer {
using ReadSource = std::variant<std::ifstream *, const char *>;

#define NN_RETURN_STATUS()                                                     \
  do {                                                                         \
    if (status != ML_ERROR_NONE) {                                             \
      return status;                                                           \
    }                                                                          \
  } while (0)

/**
 * @brief convert integer based status to throw
 *
 * @param status status to throw
 */
inline void throw_status(int status) {
  switch (status) {
  case ML_ERROR_NONE:
    break;
  case ML_ERROR_INVALID_PARAMETER:
    throw std::invalid_argument("invalid argument from c style throw");
  case ML_ERROR_OUT_OF_MEMORY:
    throw std::bad_alloc();
  case ML_ERROR_TIMED_OUT:
    throw std::runtime_error("Timed out from c style throw");
  case ML_ERROR_PERMISSION_DENIED:
    throw std::runtime_error("permission denied from c style throw");
  case ML_ERROR_UNKNOWN:
  default:
    throw std::runtime_error("unknown error from c style throw");
  }
}

static auto rng = [] {
  std::mt19937 rng;
  // rng.seed(getSeed());
  rng.seed(0);
  return rng;
}();

/**
 * @brief     sqrt function for float type
 * @param[in] x float
 */
template <typename T = float> T sqrtFloat(T x) {
  return static_cast<T>(sqrt((float)x));
}

/**
 * @brief    sqrt function for dobuld type
 *
 * @param x value to take sqrt
 * @return double return value
 */
double sqrtDouble(double x);

/**
 * @brief     abs function for float type
 * @param[in] x float
 */
template <typename T = float> T absFloat(T x) {
  return static_cast<T>(abs((float)x));
}

/**
 * @brief     log function for float type
 * @param[in] x float
 */
template <typename T = float> T logFloat(T x) {
  return static_cast<T>(log(x + 1.0e-20));
}

/**
 * @brief     exp function for float type
 * @param[in] x float
 */
template <typename T = float> T exp_util(T x) { return static_cast<T>(exp(x)); }

uint32_t ceilDiv(uint32_t a, uint32_t b);

uint32_t align(uint32_t a, uint32_t b);

#ifdef _WIN32
#ifdef _Float16
template <> _Float16 exp_util<_Float16>(_Float16 x) {
  return static_cast<_Float16>(std::exp(static_cast<float>(x)));
}
#endif
#endif

/**
 * @brief     Check if float is not nan and not inf
 * @param[in] value float
 * @note      We can switch to use std::isfinite once it will start support half
 floats
 */
template <typename T> bool isFloatValid(const T value) {
  return !((value != value) ||
           (value == std::numeric_limits<float>::infinity()) ||
           (value == -std::numeric_limits<float>::infinity()));
}

/**
 * @brief     Check Existance of File
 * @param[in] file path of the file to be checked
 * @returns   true if file exists, else false
 */
bool isFileExist(std::string file);

constexpr const char *default_error_msg =
  "[util::checkeFile] file operation failed";

/**
 * @brief same as file.read except it checks if fail to read the file
 *
 * @param file file to read
 * @param array char * array
 * @param size size of the array
 * @param error_msg error msg to print when operation fail
 * @throw std::runtime_error if file.fail() is true after read.
 */
void checkedRead(std::ifstream &file, char *array, std::streamsize size,
                 const char *error_msg = default_error_msg,
                 size_t start_offset = 0, bool read_from_offset = false);

/**
 * @brief same as file.read except it checks if fail to read the file
 *
 * @param ReadSource Source to read
 * @param array char * array
 * @param size size of the array
 * @param error_msg error msg to print when operation fail
 * @throw std::runtime_error if file.fail() is true after read.
 */
void checkedRead(ReadSource src, char *array, std::streamsize size,
                 const char *error_msg, size_t start_offset,
                 bool read_from_offset);

#if !defined(_WIN32)
/**
 * @brief One read-only mapping of a model file, shared by every load worker,
 *        whose resident size is bounded by construction.
 *
 * WHAT THIS REPLACES. The inference loaders used to mmap the WHOLE model file
 * once per graph node, on up to eight workers, and munmap it when that node's
 * read finished. Two costs: the address space is multiplied by the worker
 * count (large models hit the per-process VA and map-count limits on Android,
 * and the readahead advice is re-issued per node), and the staging peak tracks
 * the largest single weight record, because every page a worker touched stays
 * resident until its own munmap. Measured on a 3.2 GB package whose per-layer
 * embedding is one 1838 MB record: the process's own reported peak was
 * 2569 MB, of which ~1.6 GB was source-file staging.
 *
 * WHY NODE GRANULARITY IS NOT ENOUGH. "Drop the node's range when the node's
 * read ends" is exactly what the per-node munmap already did, and 2569 MB is
 * what it measures. A bound that is stated per record cannot bound a model
 * whose largest record is 1838 MB. The bound has to hold WITHIN one record.
 *
 * HOW THIS BOUNDS IT. The single mmap -> weight-pool copy is checkedRead()'s
 * `const char *` branch, one file below this declaration. While a copy is
 * running inside the registered window it advances in chunks and releases each
 * whole page STRICTLY BEHIND the copy cursor with madvise(MADV_DONTNEED).
 * Therefore:
 *
 *   - staging residency <= concurrency * chunk, for every model, independent
 *     of any record's size;
 *   - every source page is faulted exactly once and released exactly once --
 *     there is no re-fault, because nothing behind the cursor is read again;
 *   - no sampling thread, no mincore sweep, no threshold to tune.
 *
 * A page is released only when it lies wholly behind the cursor, so a page
 * that also holds bytes of the neighbouring record -- which another worker may
 * be reading right now -- is never touched. At most the first and last partial
 * page of each record stay resident until the whole mapping is dropped.
 *
 * WHY DROPPING IS SAFE. The mapping is PROT_READ MAP_PRIVATE over a file, so
 * it owns no private copies; MADV_DONTNEED zaps the page-table entries and the
 * next touch re-reads the same file bytes. (This is NOT true of private
 * ANONYMOUS memory, where MADV_DONTNEED hands back zero pages -- do not copy
 * this pattern to a heap buffer.)
 *
 * ONLY ONE WINDOW AT A TIME, AND IT IS KEYED ON THE POINTER. A process may
 * have several loads running at once; the first to construct one of these
 * registers the window and the others do not (they log and run without the
 * release). checkedRead therefore decides per copy, by asking whether the
 * SOURCE POINTER lies in [base, end) -- not whether some window is registered.
 * Distinct mappings never overlap, so a source belonging to a different load,
 * or to a caller-owned buffer, is outside the window by construction and takes
 * the plain memcpy. Keying it on window presence instead makes every read of
 * every other load throw, on a load worker, which is std::terminate.
 *
 * NNTR_LOAD_REAP_MB keeps its old name, units and meaning -- how many MiB of
 * source staging the load may hold resident -- and only its enforcement
 * changed, from a sampling sweep to this. It overrides the default budget; 0
 * disables the release entirely and the mapping then keeps every page it
 * touched until the load ends.
 *
 * NOTE posix_madvise(POSIX_MADV_DONTNEED) cannot be used for this: glibc
 * documents it as a no-op, so the plain-POSIX spelling silently frees nothing.
 * madvise(MADV_DONTNEED) is required.
 */
class LoaderStagingMap {
public:
  /**
   * @brief maps @a length bytes of @a fd read-only and registers the mapping
   *        as this process's load staging window
   * @param fd file descriptor of the model file, kept open by the caller
   * @param length bytes to map, normally the whole file
   * @param concurrency number of workers that will read through this mapping;
   *        the per-copy chunk is the staging budget divided by it
   * @param what human readable name of the file, used in error messages
   * @throw std::runtime_error if mmap fails. Constructing here rather than in
   *        a worker is deliberate: the failure used to be thrown out of a
   *        std::thread body with no handler, which is std::terminate.
   */
  LoaderStagingMap(int fd, size_t length, size_t concurrency, const char *what);

  /**
   * @brief releases the mapping's pages and unmaps it
   */
  ~LoaderStagingMap();

  LoaderStagingMap(const LoaderStagingMap &) = delete;
  LoaderStagingMap &operator=(const LoaderStagingMap &) = delete;

  /**
   * @brief the shared read-only view to hand to every load worker
   * @return pointer to the first mapped byte
   */
  const char *view() const { return base_; }

  /**
   * @brief mapped length
   * @return bytes mapped
   */
  size_t size() const { return length_; }

  /**
   * @brief staging budget, i.e. the source-file residency this load may hold
   * @return budget in bytes; 0 means the per-chunk release is disabled
   */
  static size_t budgetBytes();

private:
  char *base_;      /**< first mapped byte */
  size_t length_;   /**< mapped length in bytes */
  bool registered_; /**< whether this map is the registered staging window */
};
#endif

/**
 * @brief same as file.write except it checks if fail to write the file
 *
 * @param file file to write
 * @param array char * array
 * @param size size of the array
 * @param error_msg error msg to print when operation fail
 * @throw std::runtime_error if file.fail() is true after write.
 */
void checkedWrite(std::ostream &file, const char *array, std::streamsize size,
                  const char *error_msg = default_error_msg);
/**
 * @brief read string from a binary file
 *
 * @param file file to input
 * @return std::string result string
 */
std::string readString(std::ifstream &file,
                       const char *error_msg = default_error_msg);

/**
 * @brief write string to a binary file
 *
 * @param file file to write
 * @param str target string to write
 */
void writeString(std::ofstream &file, const std::string &str,
                 const char *error_msg = default_error_msg);

/**
 * @brief check if string ends with @a suffix
 *
 * @param target string to cehck
 * @param suffix check if string ends with @a suffix
 * @retval true @a target ends with @a suffix
 * @retval false @a target does not ends with @a suffix
 */
bool endswith(const std::string &target, const std::string &suffix);

/**
 * @brief     print instance info. as <Type at (address)>
 * @param[in] std::ostream &out, T&& t
 * @param[in] t pointer to the instance
 */
template <typename T,
          typename std::enable_if_t<std::is_pointer<T>::value, T> * = nullptr>
void printInstance(std::ostream &out, const T &t) {
  out << '<' << typeid(*t).name() << " at " << t << '>' << std::endl;
}

/**
 * @brief make "key=value" from key and value
 *
 * @tparam T type of a value
 * @param key key
 * @param value value
 * @return std::string with "key=value"
 */
template <typename T>
static std::string withKey(const std::string &key, const T &value) {
  std::stringstream ss;
  ss << key << "=" << value;
  return ss.str();
}

/**
 * @brief make "key=value1,value2,...valueN" from key and multiple values
 *
 * @tparam T type of a value
 * @param key key
 * @param value list of values
 * @return std::string with "key=value1,value2,...valueN"
 */
template <typename T>
static std::string withKey(const std::string &key,
                           std::initializer_list<T> value) {
  if (std::empty(value)) {
    throw std::invalid_argument("empty data cannot be converted");
  }
  std::stringstream ss;
  ss << key << "=";
  auto iter = value.begin();
  for (; iter != value.end() - 1; ++iter) {
    ss << *iter << ',';
  }
  ss << *iter;
  return ss.str();
}

/**
 * @brief creat a stream, and if !stream.good() throw appropriate error code
 * depending on @c errno
 *
 * @tparam T return type
 * @param path path
 * @param mode mode to open path
 * @return T created stream
 */
template <typename T>
T checkedOpenStream(const std::string &path, std::ios_base::openmode mode) {
  T model_file(path, mode);
  if (!model_file.good()) {
    const size_t error_buflen = 100;
    char error_buf[error_buflen];
    std::stringstream ss;
    ss << "[parseutil] requested file not opened, file path: " << path
       << " reason: " << SAFE_STRERROR(errno, error_buf, error_buflen);
    if (errno == EPERM || errno == EACCES) {
      throw nntrainer::exception::permission_denied(ss.str().c_str());
    } else {
      throw std::invalid_argument(ss.str().c_str());
    }
  }

  return model_file;
}

/**
 * @brief     parse string and return key & value
 * @param[in] input_str input string to split with '='
 * @param[out] key key
 * @param[out] value value
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
 */
int getKeyValue(const std::string &input_str, std::string &key,
                std::string &value);

/**
 * @brief     parse string and stored to int
 * @param[in] n_str number of data
 * @param[in] str string to parse
 * @param[in] value int value to stored
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
 */
int getValues(int n_str, std::string str, int *value);

/**
 * @brief     split string into vector with delimiter regex
 * @param[in] str string
 * @param[in] reg regular expression to use as delimiter
 * @retval    output string vector
 */
std::vector<std::string> split(const std::string &s, const std::regex &reg);

/**
 * @brief Cast insensitive string comparison
 *
 * @param a first string to compare
 * @param b second string to compare
 * @retval true if string is case-insensitive equal
 * @retval false if string is case-insensitive not equal
 */
bool istrequal(const std::string &a, const std::string &b);

/**
 * @brief Perform INT_LOGICAL_AND operation on enum class value
 *
 * @param e1 enum value
 * @param e2 enum value
 *
 * @return enum value after performing logical AND operation
 */
template <typename T, typename C = int>
bool enum_class_logical_and(T e1, T e2) {
  C i1 = static_cast<int>(e1);
  C i2 = static_cast<int>(e2);

  return (i1 & i2) != 0;
}

/**
 * @brief Perform INT_OR operation on enum class value
 *
 * @param e1 enum value
 * @param e2 enum value
 *
 * @return enum value after performing OR operation
 */
template <typename T, typename C = int> T enum_class_or(T e1, T e2) {
  C i1 = static_cast<int>(e1);
  C i2 = static_cast<int>(e2);

  return static_cast<T>(i1 | i2);
}

/**
 * @brief Find value in tuple by key (internal impl)
 *
 * @tparam Tuple Tuple type to search
 * @tparam ls... Tuple index sequence
 * @param t Tuple to search
 * @param key Key to find
 *
 * @return Found value or empty string
 */
template <typename Tuple, std::size_t... ls>
std::string find_in_tuple(const Tuple &t, std::index_sequence<ls...>,
                          const std::string &key) {
  std::string result = "";

  (..., ([&] {
     auto &&elem = std::get<ls>(t);
     if (strcmp(getPropKey(elem), key.c_str()) == 0) {
       result = elem.empty() ? "empty" : to_string(elem);
     }
   }()));

  return result;
}

/**
 * @brief Find value in tuple by key (user-friendly wrapper)
 *
 * @tparam Args Tuple element types
 * @param t Tuple to search
 * @param key Key to find
 *
 * @return Found value or empty string
 */
template <typename... Args>
std::string find_in_tuple(const std::tuple<Args...> &t,
                          const std::string &key) {
  return find_in_tuple(t, std::index_sequence_for<Args...>{}, key);
}

/**
 * @brief Convert a relative path into an absolute path.
 *
 * @param name relative path
 * @param resolved variable to store the result value.
 *
 * @return absolute path
 */
char *getRealpath(const char *name, char *resolved);

/**
 * @brief Get local time in tm struct format
 *
 * @param tp variable to store the result values
 *
 * @return tm struct
 */
tm *getLocaltime(tm *tp);

/**
 * @brief Create and return std::regex with the received string
 * @param str String in regular expression form
 * @return std::regex
 */
std::regex getRegex(const std::string &str);

/**
 * @brief  Convert a floating-point number into its fixed-point component (the
 * mantissa) and its exponent component.
 *
 * @param[in] input floating point to convert
 * @param[out] fixedpoint fixed-point
 * @param[out] exponent exponent
 */
void floatToFixedPointAndExponent(float input, int &fixedpoint, int &exponent);

/**
 * @brief Convert a fixed-point number and an exponent into a floating-point
 * number.
 *
 * @param[in] fixedpoint fixed-point
 * @param[in] exponent exponent
 * @return floating point result
 */
float fixedPointAndExponentToFloat(int fixedpoint, int exponent);

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __UTIL_FUNC_H__ */

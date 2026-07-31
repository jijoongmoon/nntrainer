// SPDX-License-Identifier: Apache-2.0
/**
 * @file	qs4cx_tensor.cpp
 * @date	17 June 2026
 * @brief	This is QS4CX_Tensor class for QS4CX quantized tensor.
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jaemin Shin <jaemin980311@google.com>
 * @bug		No known bugs except for NYI items
 */

#include <cpu_backend.h>
#include <env_compat.h>
#include <qs4cx_tensor.h>
#include <tensor.h>

#include <atomic>
#include <mutex>
#include <utility>
#include <vector>

namespace nntrainer {

namespace {
/**
 * @brief [init-latency L1] Is this process's QS4CX payload allocation
 *   load-destined, i.e. may allocate() hand back UNINITIALIZED memory?
 *
 * NNTR_QS4CX_HEAP_BYPASS is exactly the "self-owned weight payload about to be
 * filled from the model file" signal: Manager::requestWeights only takes the
 * bypass branch (weight_pool.request(UNMANAGED) + var->allocate()) under it,
 * and every QS4CX tensor reached that way is a weight (the only other producer
 * is the offline quantize tool, which never sets this env). Both zero passes
 * that allocate() used to do -- `new uint8_t[size()]{}` and
 * initialize()->setZero() -- are dead stores there: the payload is
 * subsequently overwritten IN FULL, either by TensorBase::read (bytes() ==
 * size() for QS4CX, since its data-type size is 1 and size() is overridden to
 * the packed nibble+scale byte count) or by copy_qs4cx (scopy over size()).
 *
 * TRIPWIRE FOR FUTURE READERS. "Uninitialized is safe" holds only while EVERY
 * reader of this payload writes all size() bytes. It is true of every reader in
 * this tree today (TensorBase::read and copy_qs4cx, both above). It is NOT
 * automatically true of a partial writer, and one such reader exists in the
 * sibling trees: the legacy on-disk QINT4 -> QS4CX transcode. There, for even
 * K, the QS4CX scale offset is `N*(K+1)/2` evaluated left-to-right
 * (= N*K/2 + N/2) while the nibble region is only N*(K/2) bytes, so an
 * N/2-byte gap sits between them that the transcode never touches; it used to
 * be zero purely by virtue of the allocation. If that read path (or any other
 * partial writer) is ever added here, it MUST call setZero() before
 * transcoding, or the gap becomes uninitialized heap.
 *
 * Escape hatch: NNTR_QS4CX_ALLOC_ZERO=1 restores the old double zero-fill.
 */
bool qs4cxAllocUninitialized() {
  static const bool v = nntr_env_on("NNTR_QS4CX_HEAP_BYPASS") &&
                        !nntr_env_on("NNTR_QS4CX_ALLOC_ZERO");
  return v;
}
} // namespace

QS4CX_Tensor::QS4CX_Tensor(std::string name_, Tformat fm) :
  TensorBase(name_, fm) {
  offset = 0;
}

QS4CX_Tensor::QS4CX_Tensor(const TensorDim &d, bool alloc_now, Initializer init,
                           std::string name) :
  TensorBase(d, false, init, name) {
  NNTR_THROW_IF(d.batch() != 1 || d.channel() != 1, std::invalid_argument)
    << "QS4CX_Tensor must be 2 dimensional tensor with batch size 1";

  if (alloc_now)
    allocate();
  offset = 0;
}

QS4CX_Tensor::QS4CX_Tensor(const TensorDim &d, const void *buf) :
  QS4CX_Tensor(d, true, Initializer::NONE, "") {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy_qs4cx(buf);
  }
}

void QS4CX_Tensor::allocate() {
  if (empty() || data)
    return;

  if (src_tensor) {
    allocateSrcTensor();
  } else {
    MemoryData *mem_data;

    // [init-latency L1] Load-destined payload: allocate UNINITIALIZED and skip
    // initialize(). See qs4cxAllocUninitialized() for why both zero passes are
    // dead stores. The page faults are NOT saved -- they move to the read()
    // that fills the buffer -- but the two full-arena writes are.
    const bool uninit = qs4cxAllocUninitialized();
    mem_data = new MemoryData(uninit ? (void *)(new uint8_t[size()])
                                     : (void *)(new uint8_t[size()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<uint8_t>();
      delete mem_data;
    });

    offset = 0;
    if (uninit)
      putData();
    else
      initialize();
  }
}

void *QS4CX_Tensor::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<uint8_t>() + offset;
}

void QS4CX_Tensor::pack() {
  if (packed_data) {
    return;
  }

  size_t opt_kernel_idx = 8;
  const size_t K = height();
  const size_t N = width();

  size_t packed_size = nntrainer::get_rhs_packed_size_qsi4cxp_qs4cxs1s0(
    N, K, opt_kernel_idx, true);
  packed_data = std::make_unique<uint8_t[]>(packed_size);

  nntrainer::rhs_pack_qsi4cxp_qs4cxs1s0(
    N, K, packed_data.get(), getData(),
    ((uint8_t *)getData()) + N * (K + 1) / 2, opt_kernel_idx, true);

  if (!packed_data) {
    throw std::runtime_error{"something wrong"};
  }
}

void *QS4CX_Tensor::getPackedData() const {
  if (!packed_data) {
    throw std::runtime_error{"pack before run model"};
  }

  return packed_data.get();
}

size_t QS4CX_Tensor::size() const {
  const size_t K = height();
  const size_t N = width();
  return N * (K + 1) / 2 + N * sizeof(float);
}

size_t QS4CX_Tensor::getMemoryBytes() const { return size() * sizeof(uint8_t); }

void *QS4CX_Tensor::getScale() const {
  if (!data)
    return nullptr;

  data->validate();

  const size_t K = height();
  const size_t N = width();

  return ((int8_t *)getData()) + N * (K + 1) / 2;
}

void QS4CX_Tensor::copy_qs4cx(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }
  scopy(size(), (uint8_t *)buf, 1, (uint8_t *)getData(), 1);
}

void QS4CX_Tensor::setZero() {
  uint8_t *data = (uint8_t *)getData();
  std::fill(data, data + size(), 0);
}

void QS4CX_Tensor::initialize() {
  if (empty() || !isAllocated())
    return;

  setZero();
  putData();
}

void QS4CX_Tensor::print(std::ostream &out) const {
  out << "data addr: " << getData() << '\n';
  out << dim;
  out << "[QS4CX data print skipped]" << std::endl;
}

QScheme QS4CX_Tensor::q_scheme() const { return QScheme::QS4CX; }

namespace {
/**
 * @brief Dropped plain-payload ranges, keyed by address span.
 *
 * Entry count is the number of int4 FC weights (order 10^2), and the vector is
 * only ever consulted once `dropped_any` is set, so a linear scan is cheaper
 * than a map and keeps the empty case free.
 */
std::mutex &qs4cx_dropped_mtx() {
  static std::mutex m;
  return m;
}
std::vector<std::pair<uintptr_t, uintptr_t>> &qs4cx_dropped_ranges() {
  static std::vector<std::pair<uintptr_t, uintptr_t>> v;
  return v;
}
std::atomic<bool> qs4cx_dropped_any{false};
} // namespace

void markQs4cxPayloadDropped(const void *base, size_t bytes) {
  if (base == nullptr || bytes == 0)
    return;
  const uintptr_t lo = reinterpret_cast<uintptr_t>(base);
  {
    std::lock_guard<std::mutex> lock(qs4cx_dropped_mtx());
    qs4cx_dropped_ranges().emplace_back(lo, lo + bytes);
  }
  // Release: the range must be visible to any thread that observes the flag.
  qs4cx_dropped_any.store(true, std::memory_order_release);
}

bool anyQs4cxPayloadDropped() {
  return qs4cx_dropped_any.load(std::memory_order_acquire);
}

bool isQs4cxPayloadDropped(const void *ptr) {
  if (!anyQs4cxPayloadDropped() || ptr == nullptr)
    return false;
  const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
  std::lock_guard<std::mutex> lock(qs4cx_dropped_mtx());
  for (const auto &r : qs4cx_dropped_ranges())
    if (p >= r.first && p < r.second)
      return true;
  return false;
}

} // namespace nntrainer

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
#include <cstdlib>
#include <qs4cx_tensor.h>
#include <tensor.h>

#include <cstring>
#include <functional>
#include <limits>
#include <vector>

#include <int4_utils.h>
#include <util_func.h>

namespace nntrainer {

namespace {
/**
 * @brief VALUE-checked env truthiness: set, non-empty, and not starting with
 *   '0', so that NNTR_QS4CX_ALLOC_ZERO=0 really is "off" rather than "present,
 *   therefore on". Kept file-local on purpose: this translation unit must
 *   compile against a tree that ships no shared env helper, and a header that
 *   only this file would consume does not belong in the public include set.
 */
bool envOn(const char *name) {
  const char *e = std::getenv(name);
  return e != nullptr && e[0] != '\0' && e[0] != '0';
}

/**
 * @brief Read a legacy on-disk int4 record (u16 qscheme header + KAI Section A
 *   or plain container) via @a do_read and transcode it losslessly to the
 *   canonical QS4CX in-memory layout (plain nibbles + fp32 scales). Shared by
 *   the std::ifstream and ReadSource read() overloads.
 */
void readLegacyQint4ToQs4cx(
  size_t N, size_t K, size_t start_offset,
  const std::function<void(char *, std::streamsize, size_t)> &do_read,
  uint8_t *out_nibbles, float *out_scales) {
  uint16_t scheme = 0;
  do_read(reinterpret_cast<char *>(&scheme), sizeof(uint16_t), start_offset);

  size_t body_bytes;
  if (scheme == static_cast<uint16_t>(QScheme::KAI_QSI4CXP_4x4x32))
    body_bytes = Int4Utils::kaiNibblePayloadBytes(N, K) + N * sizeof(uint16_t);
  else if (scheme == static_cast<uint16_t>(QScheme::PER_CHANNEL_AFFINE))
    body_bytes = Int4Utils::plainRecordPayloadBytes(N, K);
  else
    throw std::runtime_error(
      "[QS4CX_Tensor::read] unsupported legacy on-disk qscheme");

  std::vector<uint8_t> record(sizeof(uint16_t) + body_bytes);
  std::memcpy(record.data(), &scheme, sizeof(uint16_t));
  do_read(reinterpret_cast<char *>(record.data()) + sizeof(uint16_t),
          static_cast<std::streamsize>(body_bytes),
          start_offset + sizeof(uint16_t));

  // The transcode is a PARTIAL writer of the payload: it fills
  // plainNibbleBytes() = N*ceil(K/2) nibble bytes and the N fp32 scales at
  // plainScalesOffsetBytes() = N*(K+1)/2, so for even K an N/2-byte gap
  // between them is never written. That gap is part of the on-disk plain form
  // and must read as zero, which the allocation no longer guarantees (see the
  // "TRIPWIRE FOR FUTURE READERS" note above qs4cxAllocUninitialized()). Zero
  // it explicitly instead of relying on the allocator.
  const size_t nib = Int4Utils::plainNibbleBytes(N, K);
  const size_t scales_off = Int4Utils::plainScalesOffsetBytes(N, K);
  if (scales_off > nib)
    std::memset(out_nibbles + nib, 0, scales_off - nib);

  Int4Utils::readLegacyQint4RecordToQs4cx(record.data(), record.size(), N, K,
                                          out_nibbles, out_scales);
}

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
 * reader of this payload writes all size() bytes. TensorBase::read and
 * copy_qs4cx (both above) do. The legacy on-disk QINT4 -> QS4CX transcode
 * (readLegacyQint4ToQs4cx, above) does NOT: for even K the QS4CX scale offset
 * is `N*(K+1)/2` evaluated left-to-right (= N*K/2 + N/2) while the nibble
 * region is only N*(K/2) bytes, so an N/2-byte gap sits between them that the
 * transcode never touches. That reader therefore zeroes the gap itself instead
 * of relying on the allocation. Any further partial writer added here must do
 * the same, or the gap becomes uninitialized heap.
 *
 * Escape hatch: NNTR_QS4CX_ALLOC_ZERO=1 restores the old double zero-fill.
 */
bool qs4cxAllocUninitialized() {
  static const bool v =
    envOn("NNTR_QS4CX_HEAP_BYPASS") && !envOn("NNTR_QS4CX_ALLOC_ZERO");
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

void QS4CX_Tensor::read(std::ifstream &file, size_t start_offset,
                        bool read_from_offset) {
  if (start_offset == std::numeric_limits<size_t>::max())
    start_offset = file_offset;
  if (!isOnDiskLegacyQint4()) {
    TensorBase::read(file, start_offset, read_from_offset);
    return;
  }
  readLegacyQint4ToQs4cx(
    width(), height(), start_offset,
    [&](char *dst, std::streamsize n, size_t off) {
      checkedRead(file, dst, n, "[QS4CX_Tensor::read] legacy QINT4 read failed",
                  off, read_from_offset);
    },
    reinterpret_cast<uint8_t *>(getData()),
    reinterpret_cast<float *>(getScale()));
  putData();
}

void QS4CX_Tensor::read(ReadSource src, size_t start_offset,
                        bool read_from_offset) {
  if (start_offset == std::numeric_limits<size_t>::max())
    start_offset = file_offset;
  if (!isOnDiskLegacyQint4()) {
    TensorBase::read(src, start_offset, read_from_offset);
    return;
  }
  readLegacyQint4ToQs4cx(
    width(), height(), start_offset,
    [&](char *dst, std::streamsize n, size_t off) {
      checkedRead(src, dst, n, "[QS4CX_Tensor::read] legacy QINT4 read failed",
                  off, read_from_offset);
    },
    reinterpret_cast<uint8_t *>(getData()),
    reinterpret_cast<float *>(getScale()));
  putData();
}

} // namespace nntrainer

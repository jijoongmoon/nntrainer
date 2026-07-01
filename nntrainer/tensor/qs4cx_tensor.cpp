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
#include <qs4cx_tensor.h>
#include <tensor.h>

#include <cstring>
#include <functional>
#include <int4_utils.h>
#include <limits>
#include <util_func.h>
#include <vector>

namespace nntrainer {

namespace {
/**
 * @brief Read a legacy QINT4 on-disk record (u16 qscheme header + KAI Section A
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

  Int4Utils::readLegacyQint4RecordToQs4cx(record.data(), record.size(), N, K,
                                          out_nibbles, out_scales);
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

    mem_data = new MemoryData((void *)(new uint8_t[size()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<uint8_t>();
      delete mem_data;
    });

    offset = 0;
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
